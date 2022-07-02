{ pkgs ? import <nixpkgs> {}
, stdenv ? pkgs.stdenv
, withPytorchBin ? true
, withCuda ? true
} :

let
 pypkgs = pkgs.python3Packages;
 my_pytorch = if withPytorchBin then pypkgs.pytorch-bin else if withCuda then pypkgs.pytorchWithCuda else pypkgs.pytorchWithoutCuda;
 my_torchvision = pypkgs.torchvision.override {pytorch=my_pytorch;};
 my_cudatoolkit = if withPytorchBin then pkgs.cudatoolkit_11_3 else pkgs.cudatoolkit;

 mmcv = pypkgs.callPackage ./mmcv.nix {cudatoolkit=my_cudatoolkit; pytorch=my_pytorch; torchvision=my_torchvision;
                                       withCuda=withCuda; cudaArchList = pypkgs.pytorchWithCuda.cudaArchList;};
 mmdet = pypkgs.callPackage ./mmdetection.nix {mmcv = mmcv;};
 mmcls = pypkgs.callPackage ./mmclassification.nix {mmcv = mmcv;};
 mmseg = pypkgs.callPackage ./mmsegmentation.nix {mmcls = mmcls;};

 pythonEnv = (pkgs.python3.withPackages
  (ps: with ps;
    [
      filelock
      ipython
      jupyter
      line_profiler
      lxml
      numpy
      pandas
      pip
      python
      pytest
      my_pytorch
      my_torchvision
      tabulate
      tensorboardx
      tqdm

      mmdet
      mmseg
      mmcls
    ]));

 python_env_link_dir = "python-env";
 pip_env = "pip-env";
 pip_path = "${pip_env}/lib/python3.9/site-packages";

 shellHook = ''
export PATH=${pip_env}/bin:$PATH
export PIP_PREFIX=${pip_env}
export PYTHONPATH=".:${pip_path}"

if [ -d ${python_env_link_dir} ]
then
  echo "Remove old link: ${python_env_link_dir}"
  rm ${python_env_link_dir};
fi

echo Create symbolic link ${python_env_link_dir} to ${pythonEnv}
ln -s ${pythonEnv} ${python_env_link_dir}
'';

in pkgs.mkShell {
  buildInputs = [pythonEnv];
  shellHook = shellHook;
}# .env.overrideAttrs (x: { shellHook = shellHook; })
