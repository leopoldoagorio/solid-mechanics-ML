printf ("%s", program_name ());
arg_list = argv ();
for i = 1:nargin
  printf (" %s", arg_list{i});
endfor
printf ("\n");


arg_list = argv ();
if nargin < 3 
  error("At least Lx, Ly and Lz must be defined")
end
Lx = str2num(arg_list{1}) ; Ly = str2num(arg_list{2}) ; Lz = str2num(arg_list{3}) ;
disp(Lx + Ly)

