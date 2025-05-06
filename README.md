Install details





Blender package (bpy) requires python 3.11. SofaPython is usually built with 3.12. Build SofaPython like this:

$ cmake -DPython_EXECUTABLE=/home/arngorf/projects/cosirl/.venv/bin/python3 -DCMAKE_PREFIX_PATH=/home/arngorf/apps/SOFA_v24.12.00_Linux/ ..
$ make
$ make install DESTDIR=/home/arngorf/projects/cosirl/SofaPython3
