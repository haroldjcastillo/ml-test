.matrix The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector 
v = [1;2;3] 

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v 
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)

A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]

% Initialize constant s 
s = 2

% See how element-wise addition works
add_AB = A + B 

% See how element-wise subtraction works
sub_AB = A - B

% See how scalar multiplication works
mult_As = A * s

% Divide A by s
div_As = A / s

% What happens if we have a Matrix + scalar?
add_As = A + s

A = [4, 3; 6, 9]
B = [-2, 9; -5, 2]

sub_AB = A + B

A = [1, 2104; 1, 1416; 1, 1534; 1, 852]
B = [-40; .25]

A*B

A = [1, 3; 2, 4; 0 5]
B = [1, 0; 2, 3]

A*B


% Initialize random matrices A and B 
A = [1,2;4,5]
B = [1,1;0,2]

% Initialize a 2 by 2 identity matrix
I = eye(2)

% The above notation is the same as I = [1,0;0,1]

% What happens when we multiply I*A ? 
IA = I*A 

% How about A*I ? 
AI = A*I 

% Compute A*B 
AB = A*B 

% Is it equal to B*A? 
BA = B*A 

% Note that IA = AI but AB != BA

A = [2 1 8]'

rand(1)
randn(2, 3)

 % Other operations
 A = [1 2; 3 4; 5 6]
 A(:, 1) % means every element along that row/column
 A(:, 2) = [10, 11, 12]
 A(:) % put all elements of A into a single vector
 
 B = [11 12; 13 14; 15 16]
 C = [A B] % or C = [A, B]
 
 