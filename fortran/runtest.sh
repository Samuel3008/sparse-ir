python3 ../src/sparse_ir/dump.py 1e+4 1e-10 ir_nlambda4_ndigit10.dat
python3 mk_preset.py > sparse_ir_preset.f90 
make test