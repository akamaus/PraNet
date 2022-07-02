{ lib, fetchFromGitHub,
  withTests ? false,
  buildPythonPackage, pytest, matplotlib, mmcv, numpy, packaging}:

# with import <nixpkgs> {};
# with python3Packages;

buildPythonPackage rec {
  pname = "mmcls";
  rev = "v0.23.1";
  name = "${pname}-${rev}";

  src = fetchFromGitHub {
    owner = "open-mmlab";
    repo = "mmclassification";
    inherit rev;
    sha256 = "026av4mcr7zs7s6j1h8d1x02mzwwlb6l14rm0zajn4fyww97hls2";
  };

  buildInputs = [ pytest ];

  propagatedBuildInputs = [ numpy mmcv matplotlib packaging ];

  doCheck = withTests;

  meta = {
    description = "MMClassification is an open source image classification toolbox based on PyTorch. It is a part of the OpenMMLab project.";
    homepage = https://github.com/open-mmlab/mmclassification;
    maintainers = with lib.maintainers; [ akamaus ];
  };
}
