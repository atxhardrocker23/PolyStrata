@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;G:\Strata-3060\.venv\Lib\site-packages\cmake\data\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;%PATH%"

cd /d "G:\PolyStrata"
cmake -B "build-poly" -S "G:\PolyStrata" -DSTRATA_MMQ_TAB=ON -DCMAKE_CUDA_ARCHITECTURES=86
if errorlevel 1 exit /b %errorlevel%

cmake --build "build-poly" --target strata-poly -j 6
if errorlevel 1 exit /b %errorlevel%

echo Build successful!
