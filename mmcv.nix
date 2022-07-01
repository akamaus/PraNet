{lib, symlinkJoin, fetchFromGitHub, cudatoolkit, which, ninja,
  withTests ? false, withCuda ? true, cudaArchList,
   buildPythonPackage, addict, opencv3, pillow, pybind11, pytest-runner, pyyaml, packaging,
   yapf, terminaltables, numpy, protobuf, six, pytest, pytorch, torchvision }:

# with import <nixpkgs> {};
# with python3Packages;

let
  cudatoolkit_joined = symlinkJoin {
    name = "${cudatoolkit.name}-unsplit";
    paths = [ cudatoolkit.out cudatoolkit.lib ];
  };

in buildPythonPackage rec {
  pname = "mmcv-full";
  rev = "v1.5.2";
  name = "${pname}-${rev}";

  src = fetchFromGitHub {
    owner = "open-mmlab";
    repo = "mmcv";
    inherit rev;
    sha256 = "0xn0y006ar465zm4ysg41phih8lkh735n22jr7fzrr7bda1xizp2";
  };

  MAX_JOBS = 12;
  MMCV_WITH_OPS=1;

  FORCE_CUDA=if withCuda then 1 else 0;
  TORCH_CUDA_ARCH_LIST = lib.optionalString withCuda lib.strings.concatStringsSep ";" cudaArchList;

  nativeBuildInputs = [ninja which] ++ (if withCuda then [cudatoolkit_joined] else []);
  buildInputs = [pybind11 pytest-runner pillow pytest torchvision];

  propagatedBuildInputs = [ addict numpy packaging pytorch pyyaml opencv3 yapf terminaltables];

  doCheck = withTests;

  meta = {
    description = "base package for mm stuff";
    homepage = https://github.com/open-mmlab/mmcv;
    maintainers = with lib.maintainers; [ akamaus ];
  };
}
