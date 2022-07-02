{lib, fetchFromGitHub,
  withTests ? false,
   buildPythonPackage, prettytable, pytest, matplotlib, mmcls, numpy, cython, pycocotools, terminaltables}:

buildPythonPackage rec {
  pname = "mmseg";
  rev = "v0.26.0";
  name = "${pname}-${rev}";

  src = fetchFromGitHub {
    owner = "open-mmlab";
    repo = "mmsegmentation";
    inherit rev;
    sha256 = "1sgx9khm51czn1pl5f71hlqm9p3w551afk6la6a2r6plihw95xlf";
  };

  buildInputs = [ pytest ];

  propagatedBuildInputs = [ numpy mmcls matplotlib prettytable ];

  doCheck = withTests;

  meta = {
    description = "MMSegmentation is an open source semantic segmentation toolbox based on PyTorch. It is a part of the OpenMMLab project.";
    homepage = https://github.com/open-mmlab/mmcv;
    maintainers = with lib.maintainers; [ akamaus ];
  };
}
