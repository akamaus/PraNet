import logging
import math
import signal
import sys
from datetime import datetime
import multiprocessing as mp
from glob import glob
from pathlib import Path
import os
import re
from typing import List, Generator, Tuple, Iterator, Dict

import gpxpy
import http.server as hs
from ipywidgets import HTML
import numpy as np
import ipyleaflet as IPL
from tqdm import tqdm

from PIL import Image
import PIL.ExifTags


def load_track(path):
    with open(path, 'r') as f:
        t = gpxpy.parse(f)
        assert len(t.tracks) == 1
        return t.tracks[0]


def extract_tags_from_exif(exif):
    tags = {}
    unknown = {}

    for k, v in exif.items():
        if k in PIL.ExifTags.TAGS:
            t = PIL.ExifTags.TAGS[k]
            tags[t] = v
        else:
            unknown[k] = v
    return tags, unknown


def parse_gps_tag(tags):
    def dms_to_float(t):
        return t[0] + t[1] / 60 + t[2] / 3600

    if 'GPSInfo' in tags:
        g = tags['GPSInfo']
        if g[1] == g[1] == 'N' and g[3] == 'E':
            lat = dms_to_float(g[2])
            long = dms_to_float(g[4])
            return lat, long
        else:
            logging.warning(f'Strange GPSInfo: {g}')


def parse_datetime(tags):
    dt = tags.get('DateTime') or tags.get('DateTimeOriginal')
    if dt is None:
        raise ValueError('Undated photo', tags)
    t = datetime.strptime(dt, '%Y:%m:%d %H:%M:%S')
    return t


def iter_points(segs):
    if isinstance(segs, gpxpy.gpx.GPXTrack):
        segs = segs.segments
    for s in segs:
        for p in s.points:
            yield p


class PhotoLibrary:
    """ Object accumulating html snippets located on map, nearby snippets are glued together """
    def __init__(self, resolution=1e-3):
        self.snippet_map = {}
        self.resolution = resolution

    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def add_snippet(self, pos: tuple, snippet: HTML):
        assert isinstance(pos, tuple)
        assert isinstance(snippet, HTML)

        min_d = float('inf')
        cur_p = None
        for p in self.snippet_map:
            d = self.dist(pos, p)
            if d < self.resolution and d < min_d:
                min_d = d
                cur_p = p

        if cur_p:
            self.snippet_map[cur_p].append(snippet)
        else:
            self.snippet_map[pos] = [snippet]

    def iter_all_snippets(self) -> Iterator[HTML]:
        for p, snips in self.snippet_map.items():
            txt = ""
            for k, s in enumerate(snips):
                txt += "<div>" + s.value + "</div>"
                if k == 1:
                    break
            if k > 0:
                txt = f"<h3>{k+1} photos:</h3>" + '<div style="">' + txt + '</div>' # display: grid;grid-template-columns: repeat(4, 1fr); grid-gap: 15px;
            pop = IPL.Popup(child=HTML(txt), max_width=1280, max_height=1000)
            yield IPL.CircleMarker(location=p, popup=pop, radius=round(5 + len(snips) ** 0.5), opacity=0.5)


class TimestampCorrection:
    """ Object describing time corrections to be applied to some groups of files """
    def __init__(self, name_regex, verbatim_date, real_date):
        self.name_regex = name_regex
        if isinstance(verbatim_date, str):
            vdate = datetime.strptime(verbatim_date, '%Y-%m-%d %H:%M:%S')
        else:
            vdate = verbatim_date

        if isinstance(real_date, str):
            rdate = datetime.strptime(real_date, '%Y-%m-%d %H:%M:%S')
        else:
            rdate = real_date
        self.correction = rdate - vdate

    def should_apply(self, path):
        return re.match(self.name_regex, str(path))

    def apply(self, date):
        return date + self.correction


class PhotoTrackProcessor:
    SERVER_ADDRESS = ('127.0.0.1', 8010)

    time_corrections : Dict[str, TimestampCorrection]
    photolib: PhotoLibrary

    def __init__(self, track_pattern, photo_dirs=None, resolution=1e-3):
        self.tracks = []
        for track_path in sorted(glob(track_pattern)):
            self.tracks.append(load_track(track_path))

        if photo_dirs is None:
            photo_dirs = []
        elif not isinstance(photo_dirs, list):
            photo_dirs = [photo_dirs]
        self.photo_dirs = photo_dirs

        self.time_corrections = {}
        self.photolib = PhotoLibrary(resolution=resolution)

        self.base_photo_dir = ""
        self.server_proc = None

    @staticmethod
    def _sigterm_handler(_signo, _stack_frame):
        sys.exit(0)

    def stop_server(self):
        self.server_proc.terminate()

    def run_server(self, base_photo_dir):
        """ Need to run server in order to access photos from browser page """

        def run():
            signal.signal(signal.SIGTERM, self._sigterm_handler)
            os.chdir(base_photo_dir)
            httpd = hs.HTTPServer(self.SERVER_ADDRESS, hs.SimpleHTTPRequestHandler)
            try:
                httpd.serve_forever()
            finally:
                httpd.server_close()

        p = mp.Process(target=run)
        p.start()
        self.server_proc = p
        self.base_photo_dir = base_photo_dir

    @staticmethod
    def find_jpegs(path):
        img_paths = list(path.rglob('*.jpg')) + list(path.rglob('*.JPG'))
        return img_paths

    @staticmethod
    def get_bbox(seg):
        lats = list(map(lambda p: p.latitude, seg.points))
        longs = list(map(lambda p: p.longitude, seg.points))
        return (min(lats), min(longs)), (max(lats), max(longs))

    @classmethod
    def _locate_date(cls, tracks, date) -> "Point":
        ts = date.timestamp()
        min_dt = float('inf')
        nearest_p = None
        for p in iter_points(cls.iterate_tracks_segments(tracks)):
            dt = abs(p.time.timestamp() - ts)
            if dt < min_dt:
                min_dt = dt
                nearest_p = p
        return nearest_p, min_dt

    @classmethod
    def iterate_tracks_segments(cls, tracks):
        if not isinstance(tracks, list):
            tracks = [tracks]

        for t in tracks:
            for s in t.segments:
                yield s

    def _read_tags(self, img_paths):
        all_tags = []
        none_exif = []

        for fname in tqdm(img_paths):
            img = Image.open(fname)
            exif = img._getexif()
            if exif is None:
                none_exif.append(fname)
                continue

            tags, unknown = extract_tags_from_exif(exif)
            long_tags = []
            for t, v in tags.items():
                if isinstance(v, (str, bytes)) and len(v) > 50:
                    long_tags.append(t)

            for t in long_tags:
                tags.pop(t)

            all_tags.append({'fname': fname, **tags})

        return all_tags

    def drawBoundsMap(self, tracks=None):
        m = IPL.Map(center=(55.69, 37.72), zoom=5)
        m.layout.height = '800px'

        if tracks is None:
            tracks = self.tracks

        for seg in self.iterate_tracks_segments(tracks):
            bbox = self.get_bbox(seg)
            l = IPL.Rectangle(bounds=bbox)
            m.add_layer(l)
        return m

    def drawTrackMap(self, seg_predicate=None, zoom=8):
        if seg_predicate is None:
            seg_predicate = lambda _: True

        good_segs = list(filter(lambda seg: seg_predicate(seg), self.iterate_tracks_segments(self.tracks)))

        locs = [(p.latitude, p.longitude) for p in iter_points(good_segs)]
        arr = np.array(locs)
        m_arr = arr.mean(axis=0)

        l = IPL.Polyline(locations=[(p.latitude, p.longitude) for p in iter_points(good_segs)],
                     color='red', fill=False, weight=2)

        m = IPL.Map(center=list(m_arr), zoom=zoom)
        m.layout.height = '800px'

        m.add_layer(l)

        for ps, color in zip(self.photo_dirs, "blue orange green red violet".split()):
            jpg_paths = self.find_jpegs(ps)
            self.draw_photos(jpg_paths, color=color)

        for marker in self.photolib.iter_all_snippets():
            m.add_layer(marker)
        return m

    def draw_photos(self, paths, color='blue'):
        photos = self._read_tags(paths)

        for p_tags in photos:
            fn = p_tags['fname']
            gps_coords = None
            try:
                gps_coords = parse_gps_tag(p_tags)
            except KeyError:
                print('skipping gpxinfo', p_tags['GPSInfo'])

            corrected = False
            """ was time corrected or not """
            p = None
            """ point on track"""
            dt = 0
            """ time mismatch between photo and track point """
            try:
                pd = parse_datetime(p_tags)
            except ValueError as e:
                logging.warning(f'Skipping photo {fn}, {e}')
            if gps_coords:
                location = (gps_coords[0], gps_coords[1])
            else:
                if pd is None:
                    print(f'Warning, cant date "{fn}", skipping')
                    continue

                for tcorr in self.time_corrections.values():
                    if tcorr.should_apply(fn):
                        pd = tcorr.apply(pd)
                        corrected = True
                        print('Correcting', fn)

                p, dt = self._locate_date(self.tracks, pd)

                location = (p.latitude, p.longitude)
                if dt > 1000:
                    print('Date mismatch', p, 'dt', dt, 'exif date', pd, fn)

            if p:
                time_str = f"track_t: {p.time}"
            else:
                time_str = f"photo_t: {pd}"

            if corrected:
                time_str += '(corr)'

            html = HTML()
            fpart = str(fn)[len(str(self.base_photo_dir))+1:]
            html.value = f'<img src="http://{self.SERVER_ADDRESS[0]}:{self.SERVER_ADDRESS[1]}/{fpart}" width="297" height="222"></img>' \
                         f'<div>{time_str}, dt:{dt}</div>' \
                         f'<div>{fn}</div>' \
                         f'<a href="{fn}"> link </a>'

            self.photolib.add_snippet(location, html)

    def add_time_correction(self, tcorr: TimestampCorrection):
        self.time_corrections[tcorr.name_regex] = tcorr