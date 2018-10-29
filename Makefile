EXECUTABLE = volume_deform.out
OBJS = build/mLibSource.o build/main.o build/MarchingCubes.o build/TSDFVolume.o build/MC_table.o build/CombinedSolver.o 

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  LFLAGS += -L$(HOME)/Optlang/Opt/examples/external/OpenMesh/lib/osx -Wl,-rpath,$(HOME)/Optlang/Opt/examples/external/OpenMesh/lib/osx
endif

ifeq ($(UNAME), Linux)
  LFLAGS += -L$(HOME)/Optlang/Opt/examples/external/OpenMesh/lib/linux -Wl,-rpath,$(HOME)/Optlang/Opt/examples/external/OpenMesh/lib/linux 
endif

LFLAGS += -lOpenMeshCore -lOpenMeshTools 

include make_template.inc
