��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.16.22v2.16.1-19-g810f233968c8��
�
adam/dense_5_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_5_bias_velocity/*
dtype0*
shape:*+
shared_nameadam/dense_5_bias_velocity
�
.adam/dense_5_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_5_bias_velocity*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpadam/dense_5_bias_velocity*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
adam/dense_5_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_5_bias_momentum/*
dtype0*
shape:*+
shared_nameadam/dense_5_bias_momentum
�
.adam/dense_5_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_5_bias_momentum*
_output_shapes
:*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpadam/dense_5_bias_momentum*
_class
loc:@Variable_1*
_output_shapes
:*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0
�
adam/dense_5_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_5_kernel_velocity/*
dtype0*
shape:	�*-
shared_nameadam/dense_5_kernel_velocity
�
0adam/dense_5_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_5_kernel_velocity*
_output_shapes
:	�*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOpadam/dense_5_kernel_velocity*
_class
loc:@Variable_2*
_output_shapes
:	�*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape:	�*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
j
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:	�*
dtype0
�
adam/dense_5_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_5_kernel_momentum/*
dtype0*
shape:	�*-
shared_nameadam/dense_5_kernel_momentum
�
0adam/dense_5_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_5_kernel_momentum*
_output_shapes
:	�*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpadam/dense_5_kernel_momentum*
_class
loc:@Variable_3*
_output_shapes
:	�*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:	�*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
j
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:	�*
dtype0
�
adam/dense_4_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_4_bias_velocity/*
dtype0*
shape:�*+
shared_nameadam/dense_4_bias_velocity
�
.adam/dense_4_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_4_bias_velocity*
_output_shapes	
:�*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpadam/dense_4_bias_velocity*
_class
loc:@Variable_4*
_output_shapes	
:�*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:�*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
f
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes	
:�*
dtype0
�
adam/dense_4_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_4_bias_momentum/*
dtype0*
shape:�*+
shared_nameadam/dense_4_bias_momentum
�
.adam/dense_4_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_4_bias_momentum*
_output_shapes	
:�*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOpadam/dense_4_bias_momentum*
_class
loc:@Variable_5*
_output_shapes	
:�*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:�*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
f
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes	
:�*
dtype0
�
adam/dense_4_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_4_kernel_velocity/*
dtype0*
shape:
��*-
shared_nameadam/dense_4_kernel_velocity
�
0adam/dense_4_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_4_kernel_velocity* 
_output_shapes
:
��*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpadam/dense_4_kernel_velocity*
_class
loc:@Variable_6* 
_output_shapes
:
��*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:
��*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
k
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6* 
_output_shapes
:
��*
dtype0
�
adam/dense_4_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_4_kernel_momentum/*
dtype0*
shape:
��*-
shared_nameadam/dense_4_kernel_momentum
�
0adam/dense_4_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_4_kernel_momentum* 
_output_shapes
:
��*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpadam/dense_4_kernel_momentum*
_class
loc:@Variable_7* 
_output_shapes
:
��*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:
��*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
k
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7* 
_output_shapes
:
��*
dtype0
�
adam/dense_3_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_3_bias_velocity/*
dtype0*
shape:�*+
shared_nameadam/dense_3_bias_velocity
�
.adam/dense_3_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_3_bias_velocity*
_output_shapes	
:�*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOpadam/dense_3_bias_velocity*
_class
loc:@Variable_8*
_output_shapes	
:�*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:�*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
f
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes	
:�*
dtype0
�
adam/dense_3_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_3_bias_momentum/*
dtype0*
shape:�*+
shared_nameadam/dense_3_bias_momentum
�
.adam/dense_3_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_3_bias_momentum*
_output_shapes	
:�*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOpadam/dense_3_bias_momentum*
_class
loc:@Variable_9*
_output_shapes	
:�*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape:�*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
f
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes	
:�*
dtype0
�
adam/dense_3_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_3_kernel_velocity/*
dtype0*
shape:	�*-
shared_nameadam/dense_3_kernel_velocity
�
0adam/dense_3_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_3_kernel_velocity*
_output_shapes
:	�*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOpadam/dense_3_kernel_velocity*
_class
loc:@Variable_10*
_output_shapes
:	�*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape:	�*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
l
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
:	�*
dtype0
�
adam/dense_3_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_3_kernel_momentum/*
dtype0*
shape:	�*-
shared_nameadam/dense_3_kernel_momentum
�
0adam/dense_3_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_3_kernel_momentum*
_output_shapes
:	�*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOpadam/dense_3_kernel_momentum*
_class
loc:@Variable_11*
_output_shapes
:	�*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape:	�*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
l
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
:	�*
dtype0
�
dense_5/biasVarHandleOp*
_output_shapes
: *

debug_namedense_5/bias/*
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOpdense_5/bias*
_class
loc:@Variable_12*
_output_shapes
:*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
g
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes
:*
dtype0
�
dense_5/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_5/kernel/*
dtype0*
shape:	�*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	�*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOpdense_5/kernel*
_class
loc:@Variable_13*
_output_shapes
:	�*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:	�*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
l
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes
:	�*
dtype0
�
dense_4/biasVarHandleOp*
_output_shapes
: *

debug_namedense_4/bias/*
dtype0*
shape:�*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:�*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOpdense_4/bias*
_class
loc:@Variable_14*
_output_shapes	
:�*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:�*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
h
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes	
:�*
dtype0
�
dense_4/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_4/kernel/*
dtype0*
shape:
��*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
��*
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOpdense_4/kernel*
_class
loc:@Variable_15* 
_output_shapes
:
��*
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape:
��*
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
m
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15* 
_output_shapes
:
��*
dtype0
�
dense_3/biasVarHandleOp*
_output_shapes
: *

debug_namedense_3/bias/*
dtype0*
shape:�*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:�*
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOpdense_3/bias*
_class
loc:@Variable_16*
_output_shapes	
:�*
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape:�*
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
h
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*
_output_shapes	
:�*
dtype0
�
dense_3/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_3/kernel/*
dtype0*
shape:	�*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	�*
dtype0
�
&Variable_17/Initializer/ReadVariableOpReadVariableOpdense_3/kernel*
_class
loc:@Variable_17*
_output_shapes
:	�*
dtype0
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0*
shape:	�*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0
l
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17*
_output_shapes
:	�*
dtype0
�
adam/learning_rateVarHandleOp*
_output_shapes
: *#

debug_nameadam/learning_rate/*
dtype0*
shape: *#
shared_nameadam/learning_rate
q
&adam/learning_rate/Read/ReadVariableOpReadVariableOpadam/learning_rate*
_output_shapes
: *
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOpadam/learning_rate*
_class
loc:@Variable_18*
_output_shapes
: *
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape: *
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
c
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*
_output_shapes
: *
dtype0
�
adam/iterationVarHandleOp*
_output_shapes
: *

debug_nameadam/iteration/*
dtype0	*
shape: *
shared_nameadam/iteration
i
"adam/iteration/Read/ReadVariableOpReadVariableOpadam/iteration*
_output_shapes
: *
dtype0	
�
&Variable_19/Initializer/ReadVariableOpReadVariableOpadam/iteration*
_class
loc:@Variable_19*
_output_shapes
: *
dtype0	
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0	*
shape: *
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0	
c
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*
_output_shapes
: *
dtype0	
s
serving_default_examplesPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_examplesdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *A
f<R:
8__inference_signature_wrapper_serve_tf_examples_fn_95211

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
	output_names

	optimizer
_default_save_signature

signatures*
* 
* 
* 
* 
* 
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
* 
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations
_learning_rate

_momentums
_velocities*

trace_0* 

serving_default* 
]
_inbound_nodes
_outbound_nodes
 _losses
!	_loss_ids
"_losses_override* 
]
#_inbound_nodes
$_outbound_nodes
%_losses
&	_loss_ids
'_losses_override* 
]
(_inbound_nodes
)_outbound_nodes
*_losses
+	_loss_ids
,_losses_override* 
]
-_inbound_nodes
._outbound_nodes
/_losses
0	_loss_ids
1_losses_override* 
u
2_inbound_nodes
3_outbound_nodes
4_losses
5	_loss_ids
6_losses_override
7_build_shapes_dict* 
�
8_kernel
9bias
:_inbound_nodes
;_outbound_nodes
<_losses
=	_loss_ids
>_losses_override
?_build_shapes_dict*
�
@_kernel
Abias
B_inbound_nodes
C_outbound_nodes
D_losses
E	_loss_ids
F_losses_override
G_build_shapes_dict*
�
H_kernel
Ibias
J_inbound_nodes
K_outbound_nodes
L_losses
M	_loss_ids
N_losses_override
O_build_shapes_dict*
j
0
1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
Z12
[13*
.
80
91
@2
A3
H4
I5*
* 
UO
VARIABLE_VALUEVariable_190optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEVariable_183optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_170_operations/5/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_16-_operations/5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_150_operations/6/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_14-_operations/6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_130_operations/7/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_12-_operations/7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
VP
VARIABLE_VALUEVariable_111optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_101optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_91optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_81optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_71optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_61optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_51optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_41optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_32optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_22optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_12optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEVariable2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableConst*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_95462
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_95531��
�\
�
!__inference__traced_restore_95531
file_prefix&
assignvariableop_variable_19:	 (
assignvariableop_1_variable_18: 1
assignvariableop_2_variable_17:	�-
assignvariableop_3_variable_16:	�2
assignvariableop_4_variable_15:
��-
assignvariableop_5_variable_14:	�1
assignvariableop_6_variable_13:	�,
assignvariableop_7_variable_12:1
assignvariableop_8_variable_11:	�1
assignvariableop_9_variable_10:	�-
assignvariableop_10_variable_9:	�-
assignvariableop_11_variable_8:	�2
assignvariableop_12_variable_7:
��2
assignvariableop_13_variable_6:
��-
assignvariableop_14_variable_5:	�-
assignvariableop_15_variable_4:	�1
assignvariableop_16_variable_3:	�1
assignvariableop_17_variable_2:	�,
assignvariableop_18_variable_1:*
assignvariableop_19_variable:
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0_operations/5/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/5/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/6/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/6/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/7/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/7/bias/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_19Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_18Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_17Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_16Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_15Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_14Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_13Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_12Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_11Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_10Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_9Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_8Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_7Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_6Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_5Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_4Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_3Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_2Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_1Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variableIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_21Identity_21:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_12:+	'
%
_user_specified_nameVariable_11:+
'
%
_user_specified_nameVariable_10:*&
$
_user_specified_name
Variable_9:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_1:($
"
_user_specified_name
Variable
�

�
8__inference_signature_wrapper_serve_tf_examples_fn_95211
examples
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_serve_tf_examples_fn_95193o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:���������
"
_user_specified_name
examples:%!

_user_specified_name95197:%!

_user_specified_name95199:%!

_user_specified_name95201:%!

_user_specified_name95203:%!

_user_specified_name95205:%!

_user_specified_name95207
�)
�
!__inference_serving_default_95240

inputs
inputs_2
inputs_3
inputs_1H
5functional_1_1_dense_3_1_cast_readvariableop_resource:	�G
8functional_1_1_dense_3_1_biasadd_readvariableop_resource:	�I
5functional_1_1_dense_4_1_cast_readvariableop_resource:
��G
8functional_1_1_dense_4_1_biasadd_readvariableop_resource:	�H
5functional_1_1_dense_5_1_cast_readvariableop_resource:	�B
4functional_1_1_dense_5_1_add_readvariableop_resource:
identity��/functional_1_1/dense_3_1/BiasAdd/ReadVariableOp�,functional_1_1/dense_3_1/Cast/ReadVariableOp�/functional_1_1/dense_4_1/BiasAdd/ReadVariableOp�,functional_1_1/dense_4_1/Cast/ReadVariableOp�+functional_1_1/dense_5_1/Add/ReadVariableOp�,functional_1_1/dense_5_1/Cast/ReadVariableOpu
*functional_1_1/concatenate_1_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
%functional_1_1/concatenate_1_1/concatConcatV2inputsinputs_1inputs_2inputs_33functional_1_1/concatenate_1_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
,functional_1_1/dense_3_1/Cast/ReadVariableOpReadVariableOp5functional_1_1_dense_3_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
functional_1_1/dense_3_1/MatMulMatMul.functional_1_1/concatenate_1_1/concat:output:04functional_1_1/dense_3_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/functional_1_1/dense_3_1/BiasAdd/ReadVariableOpReadVariableOp8functional_1_1_dense_3_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 functional_1_1/dense_3_1/BiasAddBiasAdd)functional_1_1/dense_3_1/MatMul:product:07functional_1_1/dense_3_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
functional_1_1/dense_3_1/ReluRelu)functional_1_1/dense_3_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,functional_1_1/dense_4_1/Cast/ReadVariableOpReadVariableOp5functional_1_1_dense_4_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
functional_1_1/dense_4_1/MatMulMatMul+functional_1_1/dense_3_1/Relu:activations:04functional_1_1/dense_4_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/functional_1_1/dense_4_1/BiasAdd/ReadVariableOpReadVariableOp8functional_1_1_dense_4_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 functional_1_1/dense_4_1/BiasAddBiasAdd)functional_1_1/dense_4_1/MatMul:product:07functional_1_1/dense_4_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
functional_1_1/dense_4_1/ReluRelu)functional_1_1/dense_4_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,functional_1_1/dense_5_1/Cast/ReadVariableOpReadVariableOp5functional_1_1_dense_5_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
functional_1_1/dense_5_1/MatMulMatMul+functional_1_1/dense_4_1/Relu:activations:04functional_1_1/dense_5_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+functional_1_1/dense_5_1/Add/ReadVariableOpReadVariableOp4functional_1_1_dense_5_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
functional_1_1/dense_5_1/AddAddV2)functional_1_1/dense_5_1/MatMul:product:03functional_1_1/dense_5_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������o
IdentityIdentity functional_1_1/dense_5_1/Add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^functional_1_1/dense_3_1/BiasAdd/ReadVariableOp-^functional_1_1/dense_3_1/Cast/ReadVariableOp0^functional_1_1/dense_4_1/BiasAdd/ReadVariableOp-^functional_1_1/dense_4_1/Cast/ReadVariableOp,^functional_1_1/dense_5_1/Add/ReadVariableOp-^functional_1_1/dense_5_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:���������:���������:���������:���������: : : : : : 2b
/functional_1_1/dense_3_1/BiasAdd/ReadVariableOp/functional_1_1/dense_3_1/BiasAdd/ReadVariableOp2\
,functional_1_1/dense_3_1/Cast/ReadVariableOp,functional_1_1/dense_3_1/Cast/ReadVariableOp2b
/functional_1_1/dense_4_1/BiasAdd/ReadVariableOp/functional_1_1/dense_4_1/BiasAdd/ReadVariableOp2\
,functional_1_1/dense_4_1/Cast/ReadVariableOp,functional_1_1/dense_4_1/Cast/ReadVariableOp2Z
+functional_1_1/dense_5_1/Add/ReadVariableOp+functional_1_1/dense_5_1/Add/ReadVariableOp2\
,functional_1_1/dense_5_1/Cast/ReadVariableOp,functional_1_1/dense_5_1/Cast/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource
��
�
__inference__traced_save_95462
file_prefix,
"read_disablecopyonread_variable_19:	 .
$read_1_disablecopyonread_variable_18: 7
$read_2_disablecopyonread_variable_17:	�3
$read_3_disablecopyonread_variable_16:	�8
$read_4_disablecopyonread_variable_15:
��3
$read_5_disablecopyonread_variable_14:	�7
$read_6_disablecopyonread_variable_13:	�2
$read_7_disablecopyonread_variable_12:7
$read_8_disablecopyonread_variable_11:	�7
$read_9_disablecopyonread_variable_10:	�3
$read_10_disablecopyonread_variable_9:	�3
$read_11_disablecopyonread_variable_8:	�8
$read_12_disablecopyonread_variable_7:
��8
$read_13_disablecopyonread_variable_6:
��3
$read_14_disablecopyonread_variable_5:	�3
$read_15_disablecopyonread_variable_4:	�7
$read_16_disablecopyonread_variable_3:	�7
$read_17_disablecopyonread_variable_2:	�2
$read_18_disablecopyonread_variable_1:0
"read_19_disablecopyonread_variable:
savev2_const
identity_41��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_19*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_19^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0	R
IdentityIdentityRead/ReadVariableOp:value:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_18*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_18^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_17*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_17^Read_2/DisableCopyOnRead*
_output_shapes
:	�*
dtype0_

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_16*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_16^Read_3/DisableCopyOnRead*
_output_shapes	
:�*
dtype0[

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_15*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_15^Read_4/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0`

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_14*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_14^Read_5/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_13*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_13^Read_6/DisableCopyOnRead*
_output_shapes
:	�*
dtype0`
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	�i
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_variable_12*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_variable_12^Read_7/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:i
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_variable_11*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_variable_11^Read_8/DisableCopyOnRead*
_output_shapes
:	�*
dtype0`
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	�i
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_variable_10*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_variable_10^Read_9/DisableCopyOnRead*
_output_shapes
:	�*
dtype0`
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	�j
Read_10/DisableCopyOnReadDisableCopyOnRead$read_10_disablecopyonread_variable_9*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp$read_10_disablecopyonread_variable_9^Read_10/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_11/DisableCopyOnReadDisableCopyOnRead$read_11_disablecopyonread_variable_8*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp$read_11_disablecopyonread_variable_8^Read_11/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_12/DisableCopyOnReadDisableCopyOnRead$read_12_disablecopyonread_variable_7*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp$read_12_disablecopyonread_variable_7^Read_12/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��j
Read_13/DisableCopyOnReadDisableCopyOnRead$read_13_disablecopyonread_variable_6*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp$read_13_disablecopyonread_variable_6^Read_13/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��j
Read_14/DisableCopyOnReadDisableCopyOnRead$read_14_disablecopyonread_variable_5*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp$read_14_disablecopyonread_variable_5^Read_14/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_15/DisableCopyOnReadDisableCopyOnRead$read_15_disablecopyonread_variable_4*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp$read_15_disablecopyonread_variable_4^Read_15/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_16/DisableCopyOnReadDisableCopyOnRead$read_16_disablecopyonread_variable_3*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp$read_16_disablecopyonread_variable_3^Read_16/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	�j
Read_17/DisableCopyOnReadDisableCopyOnRead$read_17_disablecopyonread_variable_2*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp$read_17_disablecopyonread_variable_2^Read_17/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	�j
Read_18/DisableCopyOnReadDisableCopyOnRead$read_18_disablecopyonread_variable_1*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp$read_18_disablecopyonread_variable_1^Read_18/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:h
Read_19/DisableCopyOnReadDisableCopyOnRead"read_19_disablecopyonread_variable*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp"read_19_disablecopyonread_variable^Read_19/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0_operations/5/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/5/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/6/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/6/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/7/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/7/bias/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *#
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_40Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_41IdentityIdentity_40:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_41Identity_41:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_12:+	'
%
_user_specified_nameVariable_11:+
'
%
_user_specified_nameVariable_10:*&
$
_user_specified_name
Variable_9:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_1:($
"
_user_specified_name
Variable:=9

_output_shapes
: 

_user_specified_nameConst
�4
�
&__inference_serve_tf_examples_fn_95193
examplesH
5functional_1_1_dense_3_1_cast_readvariableop_resource:	�G
8functional_1_1_dense_3_1_biasadd_readvariableop_resource:	�I
5functional_1_1_dense_4_1_cast_readvariableop_resource:
��G
8functional_1_1_dense_4_1_biasadd_readvariableop_resource:	�H
5functional_1_1_dense_5_1_cast_readvariableop_resource:	�B
4functional_1_1_dense_5_1_add_readvariableop_resource:
identity��/functional_1_1/dense_3_1/BiasAdd/ReadVariableOp�,functional_1_1/dense_3_1/Cast/ReadVariableOp�/functional_1_1/dense_4_1/BiasAdd/ReadVariableOp�,functional_1_1/dense_4_1/Cast/ReadVariableOp�+functional_1_1/dense_5_1/Add/ReadVariableOp�,functional_1_1/dense_5_1/Cast/ReadVariableOpU
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_2Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_3Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_4Const*
_output_shapes
: *
dtype0*
valueB d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB �
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*2
value)B'BAAPLBAMZNBBRK_BBIXICBMSFTj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB �
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Const_1:output:0ParseExample/Const_2:output:0ParseExample/Const_3:output:0ParseExample/Const_4:output:0*
Tdense	
2*s
_output_shapesa
_:���������:���������:���������:���������:���������*0
dense_shapes 
:::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 u
*functional_1_1/concatenate_1_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
%functional_1_1/concatenate_1_1/concatConcatV2*ParseExample/ParseExampleV2:dense_values:0*ParseExample/ParseExampleV2:dense_values:4*ParseExample/ParseExampleV2:dense_values:1*ParseExample/ParseExampleV2:dense_values:23functional_1_1/concatenate_1_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
,functional_1_1/dense_3_1/Cast/ReadVariableOpReadVariableOp5functional_1_1_dense_3_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
functional_1_1/dense_3_1/MatMulMatMul.functional_1_1/concatenate_1_1/concat:output:04functional_1_1/dense_3_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/functional_1_1/dense_3_1/BiasAdd/ReadVariableOpReadVariableOp8functional_1_1_dense_3_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 functional_1_1/dense_3_1/BiasAddBiasAdd)functional_1_1/dense_3_1/MatMul:product:07functional_1_1/dense_3_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
functional_1_1/dense_3_1/ReluRelu)functional_1_1/dense_3_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,functional_1_1/dense_4_1/Cast/ReadVariableOpReadVariableOp5functional_1_1_dense_4_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
functional_1_1/dense_4_1/MatMulMatMul+functional_1_1/dense_3_1/Relu:activations:04functional_1_1/dense_4_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/functional_1_1/dense_4_1/BiasAdd/ReadVariableOpReadVariableOp8functional_1_1_dense_4_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 functional_1_1/dense_4_1/BiasAddBiasAdd)functional_1_1/dense_4_1/MatMul:product:07functional_1_1/dense_4_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
functional_1_1/dense_4_1/ReluRelu)functional_1_1/dense_4_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,functional_1_1/dense_5_1/Cast/ReadVariableOpReadVariableOp5functional_1_1_dense_5_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
functional_1_1/dense_5_1/MatMulMatMul+functional_1_1/dense_4_1/Relu:activations:04functional_1_1/dense_5_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+functional_1_1/dense_5_1/Add/ReadVariableOpReadVariableOp4functional_1_1_dense_5_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
functional_1_1/dense_5_1/AddAddV2)functional_1_1/dense_5_1/MatMul:product:03functional_1_1/dense_5_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������o
IdentityIdentity functional_1_1/dense_5_1/Add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^functional_1_1/dense_3_1/BiasAdd/ReadVariableOp-^functional_1_1/dense_3_1/Cast/ReadVariableOp0^functional_1_1/dense_4_1/BiasAdd/ReadVariableOp-^functional_1_1/dense_4_1/Cast/ReadVariableOp,^functional_1_1/dense_5_1/Add/ReadVariableOp-^functional_1_1/dense_5_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : : : 2b
/functional_1_1/dense_3_1/BiasAdd/ReadVariableOp/functional_1_1/dense_3_1/BiasAdd/ReadVariableOp2\
,functional_1_1/dense_3_1/Cast/ReadVariableOp,functional_1_1/dense_3_1/Cast/ReadVariableOp2b
/functional_1_1/dense_4_1/BiasAdd/ReadVariableOp/functional_1_1/dense_4_1/BiasAdd/ReadVariableOp2\
,functional_1_1/dense_4_1/Cast/ReadVariableOp,functional_1_1/dense_4_1/Cast/ReadVariableOp2Z
+functional_1_1/dense_5_1/Add/ReadVariableOp+functional_1_1/dense_5_1/Add/ReadVariableOp2\
,functional_1_1/dense_5_1/Cast/ReadVariableOp,functional_1_1/dense_5_1/Cast/ReadVariableOp:M I
#
_output_shapes
:���������
"
_user_specified_name
examples:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
examples-
serving_default_examples:0���������<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict:�.
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
	output_names

	optimizer
_default_save_signature

signatures"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations
_learning_rate

_momentums
_velocities"
_generic_user_object
�
trace_02�
!__inference_serving_default_95240�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *���
���
 
AAPL����������
 
AMZN����������
!
BRK_B����������
 
MSFT����������ztrace_0
,
serving_default"
signature_map
y
_inbound_nodes
_outbound_nodes
 _losses
!	_loss_ids
"_losses_override"
_generic_user_object
y
#_inbound_nodes
$_outbound_nodes
%_losses
&	_loss_ids
'_losses_override"
_generic_user_object
y
(_inbound_nodes
)_outbound_nodes
*_losses
+	_loss_ids
,_losses_override"
_generic_user_object
y
-_inbound_nodes
._outbound_nodes
/_losses
0	_loss_ids
1_losses_override"
_generic_user_object
�
2_inbound_nodes
3_outbound_nodes
4_losses
5	_loss_ids
6_losses_override
7_build_shapes_dict"
_generic_user_object
�
8_kernel
9bias
:_inbound_nodes
;_outbound_nodes
<_losses
=	_loss_ids
>_losses_override
?_build_shapes_dict"
_generic_user_object
�
@_kernel
Abias
B_inbound_nodes
C_outbound_nodes
D_losses
E	_loss_ids
F_losses_override
G_build_shapes_dict"
_generic_user_object
�
H_kernel
Ibias
J_inbound_nodes
K_outbound_nodes
L_losses
M	_loss_ids
N_losses_override
O_build_shapes_dict"
_generic_user_object
�
0
1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
Z12
[13"
trackable_list_wrapper
J
80
91
@2
A3
H4
I5"
trackable_list_wrapper
 "
trackable_dict_wrapper
:	 (2adam/iteration
: (2adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�B�
!__inference_serving_default_95240inputsinputs_1inputs_2inputs_3"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_signature_wrapper_serve_tf_examples_fn_95211examples"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jexamples
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
#:!	�(2dense_3/kernel
:�(2dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
$:"
��(2dense_4/kernel
:�(2dense_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
#:!	�(2dense_5/kernel
:(2dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/:-	�(2adam/dense_3_kernel_momentum
/:-	�(2adam/dense_3_kernel_velocity
):'�(2adam/dense_3_bias_momentum
):'�(2adam/dense_3_bias_velocity
0:.
��(2adam/dense_4_kernel_momentum
0:.
��(2adam/dense_4_kernel_velocity
):'�(2adam/dense_4_bias_momentum
):'�(2adam/dense_4_bias_velocity
/:-	�(2adam/dense_5_kernel_momentum
/:-	�(2adam/dense_5_kernel_velocity
(:&(2adam/dense_5_bias_momentum
(:&(2adam/dense_5_bias_velocity�
!__inference_serving_default_95240�89@AHI���
���
���
-
AAPL%�"
inputs_aapl���������
-
AMZN%�"
inputs_amzn���������
/
BRK_B&�#
inputs_brk_b���������
-
MSFT%�"
inputs_msft���������
� "!�
unknown����������
8__inference_signature_wrapper_serve_tf_examples_fn_95211x89@AHI9�6
� 
/�,
*
examples�
examples���������"3�0
.
output_0"�
output_0���������