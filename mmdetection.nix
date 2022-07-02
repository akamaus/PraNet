{lib, fetchFromGitHub,
  withTests ? false,
   buildPythonPackage, matplotlib, mmcv, numpy, cython, pycocotools, terminaltables }:

buildPythonPackage rec {
  pname = "mmdet";
  rev = "v2.24.1";
  name = "${pname}-${rev}";

  src = fetchFromGitHub {
    owner = "open-mmlab";
    repo = "mmdetection";
    inherit rev;
    sha256 = "1i8vmnlcp6dz7acjj78dvbgzsmww8m6g7hrjmwk9a3sdq0xvg4qa";
  };

  # buildInputs = [  opencv3 pybind11 pytest-runner pytest pytorch torchvision];

  propagatedBuildInputs = [ numpy mmcv cython matplotlib pycocotools terminaltables ];

  doCheck = withTests;

  meta = {
    description = "MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project.";
    homepage = https://github.com/open-mmlab/mmcv;
    maintainers = with lib.maintainers; [ akamaus ];
  };
}
