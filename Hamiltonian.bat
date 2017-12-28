:: This wrapper script is necessary to make the "Hamiltonian" command on windows work

@SET "PYTHON_EXE=%~dp0\..\python.exe"
call "%PYTHON_EXE%" "%~dp0\Hamiltonian" %*