FC = gfortran

TARGET = test_sparse_ir
OBJS = test_sparse_ir.o sparse_ir.o sparse_ir_io.o
LDLIBS = -lblas -llapack
LDFLAGS = -L/usr/lib/aarch64-linux-gnu

.SUFFIXES: .f90

%.o: %.f90
	$(COMPILE.f) $(OUTPUT_OPTION) $<

%.mod: %.f90 %.o
	@:

$(TARGET): $(OBJS)
	$(LINK.f) $^ $(LDLIBS) $(LDFLAGS) -o $@

test_sparse_ir.o: sparse_ir.mod sparse_ir_io.mod

test: $(TARGET)
	./test_sparse_ir
