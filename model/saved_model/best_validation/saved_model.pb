Â
5ß4
:
Add
x"T
y"T
z"T"
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignSub
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
ě
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
Ľ

ScatterAdd
ref"T
indices"Tindices
updates"T

output_ref"T" 
Ttype:
2	"
Tindicestype:
2	"
use_lockingbool( 
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
Á
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"train*1.11.02v1.11.0-0-gc19e29306cĹ
l
input_xPlaceholder*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ř*
shape:˙˙˙˙˙˙˙˙˙Ř
j
input_yPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:

*embedding/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@embedding*
valueB"m  @   

(embedding/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *Wa˝

(embedding/Initializer/random_uniform/maxConst*
_class
loc:@embedding*
valueB
 *Wa=*
dtype0*
_output_shapes
: 
Ý
2embedding/Initializer/random_uniform/RandomUniformRandomUniform*embedding/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	í@*

seed *
T0*
_class
loc:@embedding*
seed2 
Â
(embedding/Initializer/random_uniform/subSub(embedding/Initializer/random_uniform/max(embedding/Initializer/random_uniform/min*
T0*
_class
loc:@embedding*
_output_shapes
: 
Ő
(embedding/Initializer/random_uniform/mulMul2embedding/Initializer/random_uniform/RandomUniform(embedding/Initializer/random_uniform/sub*
T0*
_class
loc:@embedding*
_output_shapes
:	í@
Ç
$embedding/Initializer/random_uniformAdd(embedding/Initializer/random_uniform/mul(embedding/Initializer/random_uniform/min*
T0*
_class
loc:@embedding*
_output_shapes
:	í@
Ź
	embedding
VariableV2"/device:CPU:0*
shared_name *
_class
loc:@embedding*
	container *
shape:	í@*
dtype0*
_output_shapes
:	í@
Ë
embedding/AssignAssign	embedding$embedding/Initializer/random_uniform"/device:CPU:0*
validate_shape(*
_output_shapes
:	í@*
use_locking(*
T0*
_class
loc:@embedding
|
embedding/readIdentity	embedding"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
:	í@

embedding_lookup/axisConst"/device:CPU:0*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
value	B : 
Ó
embedding_lookupGatherV2embedding/readinput_xembedding_lookup/axis"/device:CPU:0*
Tparams0*
_class
loc:@embedding*,
_output_shapes
:˙˙˙˙˙˙˙˙˙Ř@*
Taxis0*
Tindices0
}
embedding_lookup/IdentityIdentityembedding_lookup"/device:CPU:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙Ř@
Ą
,conv/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@conv/kernel*!
valueB"   @      

*conv/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@conv/kernel*
valueB
 *çÓz˝

*conv/kernel/Initializer/random_uniform/maxConst*
_class
loc:@conv/kernel*
valueB
 *çÓz=*
dtype0*
_output_shapes
: 
ç
4conv/kernel/Initializer/random_uniform/RandomUniformRandomUniform,conv/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@conv/kernel*
seed2 *
dtype0*#
_output_shapes
:@
Ę
*conv/kernel/Initializer/random_uniform/subSub*conv/kernel/Initializer/random_uniform/max*conv/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@conv/kernel*
_output_shapes
: 
á
*conv/kernel/Initializer/random_uniform/mulMul4conv/kernel/Initializer/random_uniform/RandomUniform*conv/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@conv/kernel*#
_output_shapes
:@
Ó
&conv/kernel/Initializer/random_uniformAdd*conv/kernel/Initializer/random_uniform/mul*conv/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@conv/kernel*#
_output_shapes
:@
Š
conv/kernel
VariableV2*
dtype0*#
_output_shapes
:@*
shared_name *
_class
loc:@conv/kernel*
	container *
shape:@
Č
conv/kernel/AssignAssignconv/kernel&conv/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@conv/kernel*
validate_shape(*#
_output_shapes
:@
w
conv/kernel/readIdentityconv/kernel*
T0*
_class
loc:@conv/kernel*#
_output_shapes
:@

conv/bias/Initializer/zerosConst*
_class
loc:@conv/bias*
valueB*    *
dtype0*
_output_shapes	
:

	conv/bias
VariableV2*
_class
loc:@conv/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ż
conv/bias/AssignAssign	conv/biasconv/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes	
:
i
conv/bias/readIdentity	conv/bias*
T0*
_class
loc:@conv/bias*
_output_shapes	
:
`
cnn/conv/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
`
cnn/conv/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ş
cnn/conv/conv1d/ExpandDims
ExpandDimsembedding_lookup/Identitycnn/conv/conv1d/ExpandDims/dim*

Tdim0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ř@
b
 cnn/conv/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 

cnn/conv/conv1d/ExpandDims_1
ExpandDimsconv/kernel/read cnn/conv/conv1d/ExpandDims_1/dim*
T0*'
_output_shapes
:@*

Tdim0

cnn/conv/conv1d/Conv2DConv2Dcnn/conv/conv1d/ExpandDimscnn/conv/conv1d/ExpandDims_1*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ô*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

cnn/conv/conv1d/SqueezeSqueezecnn/conv/conv1d/Conv2D*
squeeze_dims
*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ô

cnn/conv/BiasAddBiasAddcnn/conv/conv1d/Squeezeconv/bias/read*
data_formatNHWC*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ô*
T0
c
cnn/gmp/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

cnn/gmpMaxcnn/conv/BiasAddcnn/gmp/reduction_indices*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0

+fc1/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@fc1/kernel*
valueB"      *
dtype0*
_output_shapes
:

)fc1/kernel/Initializer/random_uniform/minConst*
_class
loc:@fc1/kernel*
valueB
 *   ž*
dtype0*
_output_shapes
: 

)fc1/kernel/Initializer/random_uniform/maxConst*
_class
loc:@fc1/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
á
3fc1/kernel/Initializer/random_uniform/RandomUniformRandomUniform+fc1/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@fc1/kernel*
seed2 *
dtype0* 
_output_shapes
:

Ć
)fc1/kernel/Initializer/random_uniform/subSub)fc1/kernel/Initializer/random_uniform/max)fc1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@fc1/kernel
Ú
)fc1/kernel/Initializer/random_uniform/mulMul3fc1/kernel/Initializer/random_uniform/RandomUniform)fc1/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@fc1/kernel* 
_output_shapes
:

Ě
%fc1/kernel/Initializer/random_uniformAdd)fc1/kernel/Initializer/random_uniform/mul)fc1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc1/kernel* 
_output_shapes
:

Ą

fc1/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *
_class
loc:@fc1/kernel*
	container 
Á
fc1/kernel/AssignAssign
fc1/kernel%fc1/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@fc1/kernel
q
fc1/kernel/readIdentity
fc1/kernel*
T0*
_class
loc:@fc1/kernel* 
_output_shapes
:


fc1/bias/Initializer/zerosConst*
_class
loc:@fc1/bias*
valueB*    *
dtype0*
_output_shapes	
:

fc1/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@fc1/bias*
	container *
shape:
Ť
fc1/bias/AssignAssignfc1/biasfc1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc1/bias*
validate_shape(*
_output_shapes	
:
f
fc1/bias/readIdentityfc1/bias*
T0*
_class
loc:@fc1/bias*
_output_shapes	
:

score/fc1/MatMulMatMulcnn/gmpfc1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

score/fc1/BiasAddBiasAddscore/fc1/MatMulfc1/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
X
score/Dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
[
score/Dropout/subSubscore/Dropout/sub/x	keep_prob*
_output_shapes
:*
T0
Z
score/Dropout/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
g
score/Dropout/sub_1Subscore/Dropout/sub_1/xscore/Dropout/sub*
_output_shapes
:*
T0
n
score/Dropout/dropout_1/ShapeShapescore/fc1/BiasAdd*
T0*
out_type0*
_output_shapes
:
o
*score/Dropout/dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
*score/Dropout/dropout_1/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
˝
4score/Dropout/dropout_1/random_uniform/RandomUniformRandomUniformscore/Dropout/dropout_1/Shape*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2 *

seed *
T0
Ş
*score/Dropout/dropout_1/random_uniform/subSub*score/Dropout/dropout_1/random_uniform/max*score/Dropout/dropout_1/random_uniform/min*
T0*
_output_shapes
: 
Ć
*score/Dropout/dropout_1/random_uniform/mulMul4score/Dropout/dropout_1/random_uniform/RandomUniform*score/Dropout/dropout_1/random_uniform/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
&score/Dropout/dropout_1/random_uniformAdd*score/Dropout/dropout_1/random_uniform/mul*score/Dropout/dropout_1/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

score/Dropout/dropout_1/addAddscore/Dropout/sub_1&score/Dropout/dropout_1/random_uniform*
_output_shapes
:*
T0
f
score/Dropout/dropout_1/FloorFloorscore/Dropout/dropout_1/add*
T0*
_output_shapes
:
q
score/Dropout/dropout_1/divRealDivscore/fc1/BiasAddscore/Dropout/sub_1*
T0*
_output_shapes
:

score/Dropout/dropout_1/mulMulscore/Dropout/dropout_1/divscore/Dropout/dropout_1/Floor*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
b

score/ReluReluscore/Dropout/dropout_1/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

+fc2/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@fc2/kernel*
valueB"      *
dtype0*
_output_shapes
:

)fc2/kernel/Initializer/random_uniform/minConst*
_class
loc:@fc2/kernel*
valueB
 *ý[ž*
dtype0*
_output_shapes
: 

)fc2/kernel/Initializer/random_uniform/maxConst*
_class
loc:@fc2/kernel*
valueB
 *ý[>*
dtype0*
_output_shapes
: 
ŕ
3fc2/kernel/Initializer/random_uniform/RandomUniformRandomUniform+fc2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*
_class
loc:@fc2/kernel*
seed2 
Ć
)fc2/kernel/Initializer/random_uniform/subSub)fc2/kernel/Initializer/random_uniform/max)fc2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc2/kernel*
_output_shapes
: 
Ů
)fc2/kernel/Initializer/random_uniform/mulMul3fc2/kernel/Initializer/random_uniform/RandomUniform)fc2/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@fc2/kernel*
_output_shapes
:	
Ë
%fc2/kernel/Initializer/random_uniformAdd)fc2/kernel/Initializer/random_uniform/mul)fc2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc2/kernel*
_output_shapes
:	


fc2/kernel
VariableV2*
shared_name *
_class
loc:@fc2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ŕ
fc2/kernel/AssignAssign
fc2/kernel%fc2/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@fc2/kernel*
validate_shape(*
_output_shapes
:	
p
fc2/kernel/readIdentity
fc2/kernel*
T0*
_class
loc:@fc2/kernel*
_output_shapes
:	

fc2/bias/Initializer/zerosConst*
_class
loc:@fc2/bias*
valueB*    *
dtype0*
_output_shapes
:

fc2/bias
VariableV2*
_class
loc:@fc2/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ş
fc2/bias/AssignAssignfc2/biasfc2/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc2/bias*
validate_shape(*
_output_shapes
:
e
fc2/bias/readIdentityfc2/bias*
T0*
_class
loc:@fc2/bias*
_output_shapes
:

score/fc2/MatMulMatMul
score/Relufc2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

score/fc2/BiasAddBiasAddscore/fc2/MatMulfc2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z

score/probSoftmaxscore/fc2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
score/output/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

score/outputArgMax
score/probscore/output/dimension*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0

Boptimize/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientinput_y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
2optimize/softmax_cross_entropy_with_logits_sg/RankConst*
dtype0*
_output_shapes
: *
value	B :

3optimize/softmax_cross_entropy_with_logits_sg/ShapeShapescore/fc2/BiasAdd*
T0*
out_type0*
_output_shapes
:
v
4optimize/softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 

5optimize/softmax_cross_entropy_with_logits_sg/Shape_1Shapescore/fc2/BiasAdd*
T0*
out_type0*
_output_shapes
:
u
3optimize/softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ä
1optimize/softmax_cross_entropy_with_logits_sg/SubSub4optimize/softmax_cross_entropy_with_logits_sg/Rank_13optimize/softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
Ž
9optimize/softmax_cross_entropy_with_logits_sg/Slice/beginPack1optimize/softmax_cross_entropy_with_logits_sg/Sub*
N*
_output_shapes
:*
T0*

axis 

8optimize/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

3optimize/softmax_cross_entropy_with_logits_sg/SliceSlice5optimize/softmax_cross_entropy_with_logits_sg/Shape_19optimize/softmax_cross_entropy_with_logits_sg/Slice/begin8optimize/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:

=optimize/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
9optimize/softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Š
4optimize/softmax_cross_entropy_with_logits_sg/concatConcatV2=optimize/softmax_cross_entropy_with_logits_sg/concat/values_03optimize/softmax_cross_entropy_with_logits_sg/Slice9optimize/softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ň
5optimize/softmax_cross_entropy_with_logits_sg/ReshapeReshapescore/fc2/BiasAdd4optimize/softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
v
4optimize/softmax_cross_entropy_with_logits_sg/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
ˇ
5optimize/softmax_cross_entropy_with_logits_sg/Shape_2ShapeBoptimize/softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
w
5optimize/softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Č
3optimize/softmax_cross_entropy_with_logits_sg/Sub_1Sub4optimize/softmax_cross_entropy_with_logits_sg/Rank_25optimize/softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
˛
;optimize/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack3optimize/softmax_cross_entropy_with_logits_sg/Sub_1*
N*
_output_shapes
:*
T0*

axis 

:optimize/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
 
5optimize/softmax_cross_entropy_with_logits_sg/Slice_1Slice5optimize/softmax_cross_entropy_with_logits_sg/Shape_2;optimize/softmax_cross_entropy_with_logits_sg/Slice_1/begin:optimize/softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:

?optimize/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
}
;optimize/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ą
6optimize/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2?optimize/softmax_cross_entropy_with_logits_sg/concat_1/values_05optimize/softmax_cross_entropy_with_logits_sg/Slice_1;optimize/softmax_cross_entropy_with_logits_sg/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:

7optimize/softmax_cross_entropy_with_logits_sg/Reshape_1ReshapeBoptimize/softmax_cross_entropy_with_logits_sg/labels_stop_gradient6optimize/softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

-optimize/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits5optimize/softmax_cross_entropy_with_logits_sg/Reshape7optimize/softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
w
5optimize/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ć
3optimize/softmax_cross_entropy_with_logits_sg/Sub_2Sub2optimize/softmax_cross_entropy_with_logits_sg/Rank5optimize/softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 

;optimize/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
ą
:optimize/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack3optimize/softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:

5optimize/softmax_cross_entropy_with_logits_sg/Slice_2Slice3optimize/softmax_cross_entropy_with_logits_sg/Shape;optimize/softmax_cross_entropy_with_logits_sg/Slice_2/begin:optimize/softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*
_output_shapes
:
ä
7optimize/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape-optimize/softmax_cross_entropy_with_logits_sg5optimize/softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
optimize/ConstConst*
valueB: *
dtype0*
_output_shapes
:

optimize/MeanMean7optimize/softmax_cross_entropy_with_logits_sg/Reshape_2optimize/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
[
optimize/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
a
optimize/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

optimize/gradients/FillFilloptimize/gradients/Shapeoptimize/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
3optimize/gradients/optimize/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
š
-optimize/gradients/optimize/Mean_grad/ReshapeReshapeoptimize/gradients/Fill3optimize/gradients/optimize/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
˘
+optimize/gradients/optimize/Mean_grad/ShapeShape7optimize/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
Î
*optimize/gradients/optimize/Mean_grad/TileTile-optimize/gradients/optimize/Mean_grad/Reshape+optimize/gradients/optimize/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
-optimize/gradients/optimize/Mean_grad/Shape_1Shape7optimize/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
p
-optimize/gradients/optimize/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
u
+optimize/gradients/optimize/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ě
*optimize/gradients/optimize/Mean_grad/ProdProd-optimize/gradients/optimize/Mean_grad/Shape_1+optimize/gradients/optimize/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
w
-optimize/gradients/optimize/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Đ
,optimize/gradients/optimize/Mean_grad/Prod_1Prod-optimize/gradients/optimize/Mean_grad/Shape_2-optimize/gradients/optimize/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
q
/optimize/gradients/optimize/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
¸
-optimize/gradients/optimize/Mean_grad/MaximumMaximum,optimize/gradients/optimize/Mean_grad/Prod_1/optimize/gradients/optimize/Mean_grad/Maximum/y*
_output_shapes
: *
T0
ś
.optimize/gradients/optimize/Mean_grad/floordivFloorDiv*optimize/gradients/optimize/Mean_grad/Prod-optimize/gradients/optimize/Mean_grad/Maximum*
_output_shapes
: *
T0
˘
*optimize/gradients/optimize/Mean_grad/CastCast.optimize/gradients/optimize/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
ž
-optimize/gradients/optimize/Mean_grad/truedivRealDiv*optimize/gradients/optimize/Mean_grad/Tile*optimize/gradients/optimize/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
Uoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape-optimize/softmax_cross_entropy_with_logits_sg*
T0*
out_type0*
_output_shapes
:
¤
Woptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshape-optimize/gradients/optimize/Mean_grad/truedivUoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

optimize/gradients/zeros_like	ZerosLike/optimize/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Toptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
Ë
Poptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsWoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeToptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0

Ioptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/mulMulPoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/optimize/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Đ
Poptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax5optimize/softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ý
Ioptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/NegNegPoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ą
Voptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ď
Roptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsWoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeVoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Koptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/mul_1MulRoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Ioptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ř
Voptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOpJ^optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/mulL^optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/mul_1
§
^optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentityIoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/mulW^optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*\
_classR
PNloc:@optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
­
`optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1IdentityKoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/mul_1W^optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*^
_classT
RPloc:@optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/mul_1
¤
Soptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapescore/fc2/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ő
Uoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshape^optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencySoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
5optimize/gradients/score/fc2/BiasAdd_grad/BiasAddGradBiasAddGradUoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
Ň
:optimize/gradients/score/fc2/BiasAdd_grad/tuple/group_depsNoOpV^optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape6^optimize/gradients/score/fc2/BiasAdd_grad/BiasAddGrad
ţ
Boptimize/gradients/score/fc2/BiasAdd_grad/tuple/control_dependencyIdentityUoptimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape;^optimize/gradients/score/fc2/BiasAdd_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@optimize/gradients/optimize/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Doptimize/gradients/score/fc2/BiasAdd_grad/tuple/control_dependency_1Identity5optimize/gradients/score/fc2/BiasAdd_grad/BiasAddGrad;^optimize/gradients/score/fc2/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@optimize/gradients/score/fc2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ç
/optimize/gradients/score/fc2/MatMul_grad/MatMulMatMulBoptimize/gradients/score/fc2/BiasAdd_grad/tuple/control_dependencyfc2/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ű
1optimize/gradients/score/fc2/MatMul_grad/MatMul_1MatMul
score/ReluBoptimize/gradients/score/fc2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
§
9optimize/gradients/score/fc2/MatMul_grad/tuple/group_depsNoOp0^optimize/gradients/score/fc2/MatMul_grad/MatMul2^optimize/gradients/score/fc2/MatMul_grad/MatMul_1
ą
Aoptimize/gradients/score/fc2/MatMul_grad/tuple/control_dependencyIdentity/optimize/gradients/score/fc2/MatMul_grad/MatMul:^optimize/gradients/score/fc2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*B
_class8
64loc:@optimize/gradients/score/fc2/MatMul_grad/MatMul
Ž
Coptimize/gradients/score/fc2/MatMul_grad/tuple/control_dependency_1Identity1optimize/gradients/score/fc2/MatMul_grad/MatMul_1:^optimize/gradients/score/fc2/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*D
_class:
86loc:@optimize/gradients/score/fc2/MatMul_grad/MatMul_1
š
+optimize/gradients/score/Relu_grad/ReluGradReluGradAoptimize/gradients/score/fc2/MatMul_grad/tuple/control_dependency
score/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9optimize/gradients/score/Dropout/dropout_1/mul_grad/ShapeShapescore/Dropout/dropout_1/div*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
;optimize/gradients/score/Dropout/dropout_1/mul_grad/Shape_1Shapescore/Dropout/dropout_1/Floor*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ioptimize/gradients/score/Dropout/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9optimize/gradients/score/Dropout/dropout_1/mul_grad/Shape;optimize/gradients/score/Dropout/dropout_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
­
7optimize/gradients/score/Dropout/dropout_1/mul_grad/MulMul+optimize/gradients/score/Relu_grad/ReluGradscore/Dropout/dropout_1/Floor*
_output_shapes
:*
T0

7optimize/gradients/score/Dropout/dropout_1/mul_grad/SumSum7optimize/gradients/score/Dropout/dropout_1/mul_grad/MulIoptimize/gradients/score/Dropout/dropout_1/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ë
;optimize/gradients/score/Dropout/dropout_1/mul_grad/ReshapeReshape7optimize/gradients/score/Dropout/dropout_1/mul_grad/Sum9optimize/gradients/score/Dropout/dropout_1/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
­
9optimize/gradients/score/Dropout/dropout_1/mul_grad/Mul_1Mulscore/Dropout/dropout_1/div+optimize/gradients/score/Relu_grad/ReluGrad*
T0*
_output_shapes
:

9optimize/gradients/score/Dropout/dropout_1/mul_grad/Sum_1Sum9optimize/gradients/score/Dropout/dropout_1/mul_grad/Mul_1Koptimize/gradients/score/Dropout/dropout_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ń
=optimize/gradients/score/Dropout/dropout_1/mul_grad/Reshape_1Reshape9optimize/gradients/score/Dropout/dropout_1/mul_grad/Sum_1;optimize/gradients/score/Dropout/dropout_1/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
Ę
Doptimize/gradients/score/Dropout/dropout_1/mul_grad/tuple/group_depsNoOp<^optimize/gradients/score/Dropout/dropout_1/mul_grad/Reshape>^optimize/gradients/score/Dropout/dropout_1/mul_grad/Reshape_1
Ď
Loptimize/gradients/score/Dropout/dropout_1/mul_grad/tuple/control_dependencyIdentity;optimize/gradients/score/Dropout/dropout_1/mul_grad/ReshapeE^optimize/gradients/score/Dropout/dropout_1/mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@optimize/gradients/score/Dropout/dropout_1/mul_grad/Reshape*
_output_shapes
:
Ő
Noptimize/gradients/score/Dropout/dropout_1/mul_grad/tuple/control_dependency_1Identity=optimize/gradients/score/Dropout/dropout_1/mul_grad/Reshape_1E^optimize/gradients/score/Dropout/dropout_1/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*P
_classF
DBloc:@optimize/gradients/score/Dropout/dropout_1/mul_grad/Reshape_1

9optimize/gradients/score/Dropout/dropout_1/div_grad/ShapeShapescore/fc1/BiasAdd*
T0*
out_type0*
_output_shapes
:

;optimize/gradients/score/Dropout/dropout_1/div_grad/Shape_1Shapescore/Dropout/sub_1*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ioptimize/gradients/score/Dropout/dropout_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs9optimize/gradients/score/Dropout/dropout_1/div_grad/Shape;optimize/gradients/score/Dropout/dropout_1/div_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ě
;optimize/gradients/score/Dropout/dropout_1/div_grad/RealDivRealDivLoptimize/gradients/score/Dropout/dropout_1/mul_grad/tuple/control_dependencyscore/Dropout/sub_1*
T0*
_output_shapes
:

7optimize/gradients/score/Dropout/dropout_1/div_grad/SumSum;optimize/gradients/score/Dropout/dropout_1/div_grad/RealDivIoptimize/gradients/score/Dropout/dropout_1/div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ű
;optimize/gradients/score/Dropout/dropout_1/div_grad/ReshapeReshape7optimize/gradients/score/Dropout/dropout_1/div_grad/Sum9optimize/gradients/score/Dropout/dropout_1/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

7optimize/gradients/score/Dropout/dropout_1/div_grad/NegNegscore/fc1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
š
=optimize/gradients/score/Dropout/dropout_1/div_grad/RealDiv_1RealDiv7optimize/gradients/score/Dropout/dropout_1/div_grad/Negscore/Dropout/sub_1*
T0*
_output_shapes
:
ż
=optimize/gradients/score/Dropout/dropout_1/div_grad/RealDiv_2RealDiv=optimize/gradients/score/Dropout/dropout_1/div_grad/RealDiv_1score/Dropout/sub_1*
_output_shapes
:*
T0
î
7optimize/gradients/score/Dropout/dropout_1/div_grad/mulMulLoptimize/gradients/score/Dropout/dropout_1/mul_grad/tuple/control_dependency=optimize/gradients/score/Dropout/dropout_1/div_grad/RealDiv_2*
T0*
_output_shapes
:

9optimize/gradients/score/Dropout/dropout_1/div_grad/Sum_1Sum7optimize/gradients/score/Dropout/dropout_1/div_grad/mulKoptimize/gradients/score/Dropout/dropout_1/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ń
=optimize/gradients/score/Dropout/dropout_1/div_grad/Reshape_1Reshape9optimize/gradients/score/Dropout/dropout_1/div_grad/Sum_1;optimize/gradients/score/Dropout/dropout_1/div_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
Ę
Doptimize/gradients/score/Dropout/dropout_1/div_grad/tuple/group_depsNoOp<^optimize/gradients/score/Dropout/dropout_1/div_grad/Reshape>^optimize/gradients/score/Dropout/dropout_1/div_grad/Reshape_1
ß
Loptimize/gradients/score/Dropout/dropout_1/div_grad/tuple/control_dependencyIdentity;optimize/gradients/score/Dropout/dropout_1/div_grad/ReshapeE^optimize/gradients/score/Dropout/dropout_1/div_grad/tuple/group_deps*
T0*N
_classD
B@loc:@optimize/gradients/score/Dropout/dropout_1/div_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
Noptimize/gradients/score/Dropout/dropout_1/div_grad/tuple/control_dependency_1Identity=optimize/gradients/score/Dropout/dropout_1/div_grad/Reshape_1E^optimize/gradients/score/Dropout/dropout_1/div_grad/tuple/group_deps*
_output_shapes
:*
T0*P
_classF
DBloc:@optimize/gradients/score/Dropout/dropout_1/div_grad/Reshape_1
Ď
5optimize/gradients/score/fc1/BiasAdd_grad/BiasAddGradBiasAddGradLoptimize/gradients/score/Dropout/dropout_1/div_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:*
T0
É
:optimize/gradients/score/fc1/BiasAdd_grad/tuple/group_depsNoOpM^optimize/gradients/score/Dropout/dropout_1/div_grad/tuple/control_dependency6^optimize/gradients/score/fc1/BiasAdd_grad/BiasAddGrad
Ü
Boptimize/gradients/score/fc1/BiasAdd_grad/tuple/control_dependencyIdentityLoptimize/gradients/score/Dropout/dropout_1/div_grad/tuple/control_dependency;^optimize/gradients/score/fc1/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@optimize/gradients/score/Dropout/dropout_1/div_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
Doptimize/gradients/score/fc1/BiasAdd_grad/tuple/control_dependency_1Identity5optimize/gradients/score/fc1/BiasAdd_grad/BiasAddGrad;^optimize/gradients/score/fc1/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@optimize/gradients/score/fc1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ç
/optimize/gradients/score/fc1/MatMul_grad/MatMulMatMulBoptimize/gradients/score/fc1/BiasAdd_grad/tuple/control_dependencyfc1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ů
1optimize/gradients/score/fc1/MatMul_grad/MatMul_1MatMulcnn/gmpBoptimize/gradients/score/fc1/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
§
9optimize/gradients/score/fc1/MatMul_grad/tuple/group_depsNoOp0^optimize/gradients/score/fc1/MatMul_grad/MatMul2^optimize/gradients/score/fc1/MatMul_grad/MatMul_1
ą
Aoptimize/gradients/score/fc1/MatMul_grad/tuple/control_dependencyIdentity/optimize/gradients/score/fc1/MatMul_grad/MatMul:^optimize/gradients/score/fc1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@optimize/gradients/score/fc1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
Coptimize/gradients/score/fc1/MatMul_grad/tuple/control_dependency_1Identity1optimize/gradients/score/fc1/MatMul_grad/MatMul_1:^optimize/gradients/score/fc1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*D
_class:
86loc:@optimize/gradients/score/fc1/MatMul_grad/MatMul_1
u
%optimize/gradients/cnn/gmp_grad/ShapeShapecnn/conv/BiasAdd*
T0*
out_type0*
_output_shapes
:
f
$optimize/gradients/cnn/gmp_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 

#optimize/gradients/cnn/gmp_grad/addAddcnn/gmp/reduction_indices$optimize/gradients/cnn/gmp_grad/Size*
T0*
_output_shapes
:

#optimize/gradients/cnn/gmp_grad/modFloorMod#optimize/gradients/cnn/gmp_grad/add$optimize/gradients/cnn/gmp_grad/Size*
T0*
_output_shapes
:
q
'optimize/gradients/cnn/gmp_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
m
+optimize/gradients/cnn/gmp_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
m
+optimize/gradients/cnn/gmp_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ö
%optimize/gradients/cnn/gmp_grad/rangeRange+optimize/gradients/cnn/gmp_grad/range/start$optimize/gradients/cnn/gmp_grad/Size+optimize/gradients/cnn/gmp_grad/range/delta*

Tidx0*
_output_shapes
:
l
*optimize/gradients/cnn/gmp_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
¸
$optimize/gradients/cnn/gmp_grad/FillFill'optimize/gradients/cnn/gmp_grad/Shape_1*optimize/gradients/cnn/gmp_grad/Fill/value*
T0*

index_type0*
_output_shapes
:

-optimize/gradients/cnn/gmp_grad/DynamicStitchDynamicStitch%optimize/gradients/cnn/gmp_grad/range#optimize/gradients/cnn/gmp_grad/mod%optimize/gradients/cnn/gmp_grad/Shape$optimize/gradients/cnn/gmp_grad/Fill*
T0*
N*
_output_shapes
:
Ŕ
'optimize/gradients/cnn/gmp_grad/ReshapeReshapecnn/gmp-optimize/gradients/cnn/gmp_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ü
)optimize/gradients/cnn/gmp_grad/Reshape_1ReshapeAoptimize/gradients/score/fc1/MatMul_grad/tuple/control_dependency-optimize/gradients/cnn/gmp_grad/DynamicStitch*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ą
%optimize/gradients/cnn/gmp_grad/EqualEqual'optimize/gradients/cnn/gmp_grad/Reshapecnn/conv/BiasAdd*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ô
Ş
$optimize/gradients/cnn/gmp_grad/CastCast%optimize/gradients/cnn/gmp_grad/Equal*

SrcT0
*
Truncate( *-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ô*

DstT0
ť
#optimize/gradients/cnn/gmp_grad/SumSum$optimize/gradients/cnn/gmp_grad/Castcnn/gmp/reduction_indices*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0*
T0
Ţ
)optimize/gradients/cnn/gmp_grad/Reshape_2Reshape#optimize/gradients/cnn/gmp_grad/Sum-optimize/gradients/cnn/gmp_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ˇ
#optimize/gradients/cnn/gmp_grad/divRealDiv$optimize/gradients/cnn/gmp_grad/Cast)optimize/gradients/cnn/gmp_grad/Reshape_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ô
˛
#optimize/gradients/cnn/gmp_grad/mulMul#optimize/gradients/cnn/gmp_grad/div)optimize/gradients/cnn/gmp_grad/Reshape_1*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ô
Ľ
4optimize/gradients/cnn/conv/BiasAdd_grad/BiasAddGradBiasAddGrad#optimize/gradients/cnn/gmp_grad/mul*
T0*
data_formatNHWC*
_output_shapes	
:

9optimize/gradients/cnn/conv/BiasAdd_grad/tuple/group_depsNoOp5^optimize/gradients/cnn/conv/BiasAdd_grad/BiasAddGrad$^optimize/gradients/cnn/gmp_grad/mul

Aoptimize/gradients/cnn/conv/BiasAdd_grad/tuple/control_dependencyIdentity#optimize/gradients/cnn/gmp_grad/mul:^optimize/gradients/cnn/conv/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@optimize/gradients/cnn/gmp_grad/mul*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ô
°
Coptimize/gradients/cnn/conv/BiasAdd_grad/tuple/control_dependency_1Identity4optimize/gradients/cnn/conv/BiasAdd_grad/BiasAddGrad:^optimize/gradients/cnn/conv/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@optimize/gradients/cnn/conv/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

5optimize/gradients/cnn/conv/conv1d/Squeeze_grad/ShapeShapecnn/conv/conv1d/Conv2D*
T0*
out_type0*
_output_shapes
:

7optimize/gradients/cnn/conv/conv1d/Squeeze_grad/ReshapeReshapeAoptimize/gradients/cnn/conv/BiasAdd_grad/tuple/control_dependency5optimize/gradients/cnn/conv/conv1d/Squeeze_grad/Shape*
T0*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ô
˝
5optimize/gradients/cnn/conv/conv1d/Conv2D_grad/ShapeNShapeNcnn/conv/conv1d/ExpandDimscnn/conv/conv1d/ExpandDims_1*
N* 
_output_shapes
::*
T0*
out_type0

Boptimize/gradients/cnn/conv/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5optimize/gradients/cnn/conv/conv1d/Conv2D_grad/ShapeNcnn/conv/conv1d/ExpandDims_17optimize/gradients/cnn/conv/conv1d/Squeeze_grad/Reshape*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ř@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

Coptimize/gradients/cnn/conv/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltercnn/conv/conv1d/ExpandDims7optimize/gradients/cnn/conv/conv1d/Conv2D_grad/ShapeN:17optimize/gradients/cnn/conv/conv1d/Squeeze_grad/Reshape*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:@
Ň
?optimize/gradients/cnn/conv/conv1d/Conv2D_grad/tuple/group_depsNoOpD^optimize/gradients/cnn/conv/conv1d/Conv2D_grad/Conv2DBackpropFilterC^optimize/gradients/cnn/conv/conv1d/Conv2D_grad/Conv2DBackpropInput
ë
Goptimize/gradients/cnn/conv/conv1d/Conv2D_grad/tuple/control_dependencyIdentityBoptimize/gradients/cnn/conv/conv1d/Conv2D_grad/Conv2DBackpropInput@^optimize/gradients/cnn/conv/conv1d/Conv2D_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ř@*
T0*U
_classK
IGloc:@optimize/gradients/cnn/conv/conv1d/Conv2D_grad/Conv2DBackpropInput
ć
Ioptimize/gradients/cnn/conv/conv1d/Conv2D_grad/tuple/control_dependency_1IdentityCoptimize/gradients/cnn/conv/conv1d/Conv2D_grad/Conv2DBackpropFilter@^optimize/gradients/cnn/conv/conv1d/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@optimize/gradients/cnn/conv/conv1d/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@

8optimize/gradients/cnn/conv/conv1d/ExpandDims_grad/ShapeShapeembedding_lookup/Identity*
T0*
out_type0*
_output_shapes
:

:optimize/gradients/cnn/conv/conv1d/ExpandDims_grad/ReshapeReshapeGoptimize/gradients/cnn/conv/conv1d/Conv2D_grad/tuple/control_dependency8optimize/gradients/cnn/conv/conv1d/ExpandDims_grad/Shape*
T0*
Tshape0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙Ř@

:optimize/gradients/cnn/conv/conv1d/ExpandDims_1_grad/ShapeConst*
dtype0*
_output_shapes
:*!
valueB"   @      

<optimize/gradients/cnn/conv/conv1d/ExpandDims_1_grad/ReshapeReshapeIoptimize/gradients/cnn/conv/conv1d/Conv2D_grad/tuple/control_dependency_1:optimize/gradients/cnn/conv/conv1d/ExpandDims_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:@
´
.optimize/gradients/embedding_lookup_grad/ShapeConst"/device:CPU:0*
_class
loc:@embedding*%
valueB	"m      @       *
dtype0	*
_output_shapes
:
Ů
0optimize/gradients/embedding_lookup_grad/ToInt32Cast.optimize/gradients/embedding_lookup_grad/Shape"/device:CPU:0*

SrcT0	*
_class
loc:@embedding*
Truncate( *
_output_shapes
:*

DstT0
o
-optimize/gradients/embedding_lookup_grad/SizeSizeinput_x*
T0*
out_type0*
_output_shapes
: 
y
7optimize/gradients/embedding_lookup_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ú
3optimize/gradients/embedding_lookup_grad/ExpandDims
ExpandDims-optimize/gradients/embedding_lookup_grad/Size7optimize/gradients/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:

<optimize/gradients/embedding_lookup_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:

>optimize/gradients/embedding_lookup_grad/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 

>optimize/gradients/embedding_lookup_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ě
6optimize/gradients/embedding_lookup_grad/strided_sliceStridedSlice0optimize/gradients/embedding_lookup_grad/ToInt32<optimize/gradients/embedding_lookup_grad/strided_slice/stack>optimize/gradients/embedding_lookup_grad/strided_slice/stack_1>optimize/gradients/embedding_lookup_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
v
4optimize/gradients/embedding_lookup_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

/optimize/gradients/embedding_lookup_grad/concatConcatV23optimize/gradients/embedding_lookup_grad/ExpandDims6optimize/gradients/embedding_lookup_grad/strided_slice4optimize/gradients/embedding_lookup_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
č
0optimize/gradients/embedding_lookup_grad/ReshapeReshape:optimize/gradients/cnn/conv/conv1d/ExpandDims_grad/Reshape/optimize/gradients/embedding_lookup_grad/concat*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ˇ
2optimize/gradients/embedding_lookup_grad/Reshape_1Reshapeinput_x3optimize/gradients/embedding_lookup_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

"optimize/beta1_power/initial_valueConst*
_class
loc:@conv/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 

optimize/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv/bias*
	container *
shape: 
Ç
optimize/beta1_power/AssignAssignoptimize/beta1_power"optimize/beta1_power/initial_value*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes
: *
use_locking(
z
optimize/beta1_power/readIdentityoptimize/beta1_power*
T0*
_class
loc:@conv/bias*
_output_shapes
: 

"optimize/beta2_power/initial_valueConst*
_class
loc:@conv/bias*
valueB
 *wž?*
dtype0*
_output_shapes
: 

optimize/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv/bias*
	container *
shape: 
Ç
optimize/beta2_power/AssignAssignoptimize/beta2_power"optimize/beta2_power/initial_value*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes
: *
use_locking(
z
optimize/beta2_power/readIdentityoptimize/beta2_power*
T0*
_class
loc:@conv/bias*
_output_shapes
: 
Ž
0embedding/Adam/Initializer/zeros/shape_as_tensorConst"/device:CPU:0*
_class
loc:@embedding*
valueB"m  @   *
dtype0*
_output_shapes
:

&embedding/Adam/Initializer/zeros/ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *    
ë
 embedding/Adam/Initializer/zerosFill0embedding/Adam/Initializer/zeros/shape_as_tensor&embedding/Adam/Initializer/zeros/Const"/device:CPU:0*
T0*
_class
loc:@embedding*

index_type0*
_output_shapes
:	í@
ą
embedding/Adam
VariableV2"/device:CPU:0*
shape:	í@*
dtype0*
_output_shapes
:	í@*
shared_name *
_class
loc:@embedding*
	container 
Ń
embedding/Adam/AssignAssignembedding/Adam embedding/Adam/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
:	í@

embedding/Adam/readIdentityembedding/Adam"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
:	í@
°
2embedding/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:CPU:0*
dtype0*
_output_shapes
:*
_class
loc:@embedding*
valueB"m  @   

(embedding/Adam_1/Initializer/zeros/ConstConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *    *
dtype0*
_output_shapes
: 
ń
"embedding/Adam_1/Initializer/zerosFill2embedding/Adam_1/Initializer/zeros/shape_as_tensor(embedding/Adam_1/Initializer/zeros/Const"/device:CPU:0*
T0*
_class
loc:@embedding*

index_type0*
_output_shapes
:	í@
ł
embedding/Adam_1
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:	í@*
shared_name *
_class
loc:@embedding*
	container *
shape:	í@
×
embedding/Adam_1/AssignAssignembedding/Adam_1"embedding/Adam_1/Initializer/zeros"/device:CPU:0*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
:	í@*
use_locking(

embedding/Adam_1/readIdentityembedding/Adam_1"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
:	í@
§
2conv/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@conv/kernel*!
valueB"   @      

(conv/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@conv/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
č
"conv/kernel/Adam/Initializer/zerosFill2conv/kernel/Adam/Initializer/zeros/shape_as_tensor(conv/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@conv/kernel*

index_type0*#
_output_shapes
:@
Ž
conv/kernel/Adam
VariableV2*
dtype0*#
_output_shapes
:@*
shared_name *
_class
loc:@conv/kernel*
	container *
shape:@
Î
conv/kernel/Adam/AssignAssignconv/kernel/Adam"conv/kernel/Adam/Initializer/zeros*
validate_shape(*#
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv/kernel

conv/kernel/Adam/readIdentityconv/kernel/Adam*
T0*
_class
loc:@conv/kernel*#
_output_shapes
:@
Š
4conv/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@conv/kernel*!
valueB"   @      *
dtype0*
_output_shapes
:

*conv/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@conv/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
î
$conv/kernel/Adam_1/Initializer/zerosFill4conv/kernel/Adam_1/Initializer/zeros/shape_as_tensor*conv/kernel/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@conv/kernel*

index_type0*#
_output_shapes
:@
°
conv/kernel/Adam_1
VariableV2*
shape:@*
dtype0*#
_output_shapes
:@*
shared_name *
_class
loc:@conv/kernel*
	container 
Ô
conv/kernel/Adam_1/AssignAssignconv/kernel/Adam_1$conv/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv/kernel*
validate_shape(*#
_output_shapes
:@

conv/kernel/Adam_1/readIdentityconv/kernel/Adam_1*
T0*
_class
loc:@conv/kernel*#
_output_shapes
:@

 conv/bias/Adam/Initializer/zerosConst*
_class
loc:@conv/bias*
valueB*    *
dtype0*
_output_shapes	
:

conv/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@conv/bias*
	container *
shape:
ž
conv/bias/Adam/AssignAssignconv/bias/Adam conv/bias/Adam/Initializer/zeros*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
s
conv/bias/Adam/readIdentityconv/bias/Adam*
T0*
_class
loc:@conv/bias*
_output_shapes	
:

"conv/bias/Adam_1/Initializer/zerosConst*
_class
loc:@conv/bias*
valueB*    *
dtype0*
_output_shapes	
:

conv/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@conv/bias*
	container *
shape:
Ä
conv/bias/Adam_1/AssignAssignconv/bias/Adam_1"conv/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes	
:
w
conv/bias/Adam_1/readIdentityconv/bias/Adam_1*
T0*
_class
loc:@conv/bias*
_output_shapes	
:
Ą
1fc1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc1/kernel*
valueB"      *
dtype0*
_output_shapes
:

'fc1/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@fc1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
á
!fc1/kernel/Adam/Initializer/zerosFill1fc1/kernel/Adam/Initializer/zeros/shape_as_tensor'fc1/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@fc1/kernel*

index_type0* 
_output_shapes
:

Ś
fc1/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *
_class
loc:@fc1/kernel*
	container *
shape:

Ç
fc1/kernel/Adam/AssignAssignfc1/kernel/Adam!fc1/kernel/Adam/Initializer/zeros*
T0*
_class
loc:@fc1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
{
fc1/kernel/Adam/readIdentityfc1/kernel/Adam*
T0*
_class
loc:@fc1/kernel* 
_output_shapes
:

Ł
3fc1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc1/kernel*
valueB"      *
dtype0*
_output_shapes
:

)fc1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@fc1/kernel*
valueB
 *    
ç
#fc1/kernel/Adam_1/Initializer/zerosFill3fc1/kernel/Adam_1/Initializer/zeros/shape_as_tensor)fc1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*
_class
loc:@fc1/kernel*

index_type0
¨
fc1/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *
_class
loc:@fc1/kernel*
	container *
shape:

Í
fc1/kernel/Adam_1/AssignAssignfc1/kernel/Adam_1#fc1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc1/kernel*
validate_shape(* 
_output_shapes
:


fc1/kernel/Adam_1/readIdentityfc1/kernel/Adam_1*
T0*
_class
loc:@fc1/kernel* 
_output_shapes
:


fc1/bias/Adam/Initializer/zerosConst*
_class
loc:@fc1/bias*
valueB*    *
dtype0*
_output_shapes	
:

fc1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@fc1/bias*
	container *
shape:
ş
fc1/bias/Adam/AssignAssignfc1/bias/Adamfc1/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc1/bias*
validate_shape(*
_output_shapes	
:
p
fc1/bias/Adam/readIdentityfc1/bias/Adam*
T0*
_class
loc:@fc1/bias*
_output_shapes	
:

!fc1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@fc1/bias*
valueB*    

fc1/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@fc1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ŕ
fc1/bias/Adam_1/AssignAssignfc1/bias/Adam_1!fc1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc1/bias*
validate_shape(*
_output_shapes	
:
t
fc1/bias/Adam_1/readIdentityfc1/bias/Adam_1*
T0*
_class
loc:@fc1/bias*
_output_shapes	
:

!fc2/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	*
_class
loc:@fc2/kernel*
valueB	*    
¤
fc2/kernel/Adam
VariableV2*
shared_name *
_class
loc:@fc2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ć
fc2/kernel/Adam/AssignAssignfc2/kernel/Adam!fc2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc2/kernel*
validate_shape(*
_output_shapes
:	
z
fc2/kernel/Adam/readIdentityfc2/kernel/Adam*
_output_shapes
:	*
T0*
_class
loc:@fc2/kernel

#fc2/kernel/Adam_1/Initializer/zerosConst*
_class
loc:@fc2/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Ś
fc2/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@fc2/kernel*
	container *
shape:	
Ě
fc2/kernel/Adam_1/AssignAssignfc2/kernel/Adam_1#fc2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc2/kernel*
validate_shape(*
_output_shapes
:	
~
fc2/kernel/Adam_1/readIdentityfc2/kernel/Adam_1*
_output_shapes
:	*
T0*
_class
loc:@fc2/kernel

fc2/bias/Adam/Initializer/zerosConst*
_class
loc:@fc2/bias*
valueB*    *
dtype0*
_output_shapes
:

fc2/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@fc2/bias*
	container *
shape:
š
fc2/bias/Adam/AssignAssignfc2/bias/Adamfc2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@fc2/bias
o
fc2/bias/Adam/readIdentityfc2/bias/Adam*
T0*
_class
loc:@fc2/bias*
_output_shapes
:

!fc2/bias/Adam_1/Initializer/zerosConst*
_class
loc:@fc2/bias*
valueB*    *
dtype0*
_output_shapes
:

fc2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@fc2/bias*
	container *
shape:
ż
fc2/bias/Adam_1/AssignAssignfc2/bias/Adam_1!fc2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc2/bias*
validate_shape(*
_output_shapes
:
s
fc2/bias/Adam_1/readIdentityfc2/bias/Adam_1*
_output_shapes
:*
T0*
_class
loc:@fc2/bias
`
optimize/Adam/learning_rateConst*
valueB
 *ˇŃ8*
dtype0*
_output_shapes
: 
X
optimize/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
X
optimize/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wž?
Z
optimize/Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Ü
%optimize/Adam/update_embedding/UniqueUnique2optimize/gradients/embedding_lookup_grad/Reshape_1"/device:CPU:0*
out_idx0*
T0*
_class
loc:@embedding*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ś
$optimize/Adam/update_embedding/ShapeShape%optimize/Adam/update_embedding/Unique"/device:CPU:0*
T0*
_class
loc:@embedding*
out_type0*
_output_shapes
:
Š
2optimize/Adam/update_embedding/strided_slice/stackConst"/device:CPU:0*
_class
loc:@embedding*
valueB: *
dtype0*
_output_shapes
:
Ť
4optimize/Adam/update_embedding/strided_slice/stack_1Const"/device:CPU:0*
_class
loc:@embedding*
valueB:*
dtype0*
_output_shapes
:
Ť
4optimize/Adam/update_embedding/strided_slice/stack_2Const"/device:CPU:0*
_class
loc:@embedding*
valueB:*
dtype0*
_output_shapes
:
Á
,optimize/Adam/update_embedding/strided_sliceStridedSlice$optimize/Adam/update_embedding/Shape2optimize/Adam/update_embedding/strided_slice/stack4optimize/Adam/update_embedding/strided_slice/stack_14optimize/Adam/update_embedding/strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
_class
loc:@embedding
Ó
1optimize/Adam/update_embedding/UnsortedSegmentSumUnsortedSegmentSum0optimize/gradients/embedding_lookup_grad/Reshape'optimize/Adam/update_embedding/Unique:1,optimize/Adam/update_embedding/strided_slice"/device:CPU:0*
Tnumsegments0*
Tindices0*
T0*
_class
loc:@embedding*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@

$optimize/Adam/update_embedding/sub/xConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¸
"optimize/Adam/update_embedding/subSub$optimize/Adam/update_embedding/sub/xoptimize/beta2_power/read"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
: 

#optimize/Adam/update_embedding/SqrtSqrt"optimize/Adam/update_embedding/sub"/device:CPU:0*
_output_shapes
: *
T0*
_class
loc:@embedding
š
"optimize/Adam/update_embedding/mulMuloptimize/Adam/learning_rate#optimize/Adam/update_embedding/Sqrt"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
: 

&optimize/Adam/update_embedding/sub_1/xConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ź
$optimize/Adam/update_embedding/sub_1Sub&optimize/Adam/update_embedding/sub_1/xoptimize/beta1_power/read"/device:CPU:0*
_output_shapes
: *
T0*
_class
loc:@embedding
É
&optimize/Adam/update_embedding/truedivRealDiv"optimize/Adam/update_embedding/mul$optimize/Adam/update_embedding/sub_1"/device:CPU:0*
_output_shapes
: *
T0*
_class
loc:@embedding

&optimize/Adam/update_embedding/sub_2/xConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ś
$optimize/Adam/update_embedding/sub_2Sub&optimize/Adam/update_embedding/sub_2/xoptimize/Adam/beta1"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
: 
ă
$optimize/Adam/update_embedding/mul_1Mul1optimize/Adam/update_embedding/UnsortedSegmentSum$optimize/Adam/update_embedding/sub_2"/device:CPU:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0*
_class
loc:@embedding
Ź
$optimize/Adam/update_embedding/mul_2Mulembedding/Adam/readoptimize/Adam/beta1"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
:	í@
ĺ
%optimize/Adam/update_embedding/AssignAssignembedding/Adam$optimize/Adam/update_embedding/mul_2"/device:CPU:0*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
:	í@
ś
)optimize/Adam/update_embedding/ScatterAdd
ScatterAddembedding/Adam%optimize/Adam/update_embedding/Unique$optimize/Adam/update_embedding/mul_1&^optimize/Adam/update_embedding/Assign"/device:CPU:0*
use_locking( *
Tindices0*
T0*
_class
loc:@embedding*
_output_shapes
:	í@
đ
$optimize/Adam/update_embedding/mul_3Mul1optimize/Adam/update_embedding/UnsortedSegmentSum1optimize/Adam/update_embedding/UnsortedSegmentSum"/device:CPU:0*
T0*
_class
loc:@embedding*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@

&optimize/Adam/update_embedding/sub_3/xConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ś
$optimize/Adam/update_embedding/sub_3Sub&optimize/Adam/update_embedding/sub_3/xoptimize/Adam/beta2"/device:CPU:0*
_output_shapes
: *
T0*
_class
loc:@embedding
Ö
$optimize/Adam/update_embedding/mul_4Mul$optimize/Adam/update_embedding/mul_3$optimize/Adam/update_embedding/sub_3"/device:CPU:0*
T0*
_class
loc:@embedding*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Ž
$optimize/Adam/update_embedding/mul_5Mulembedding/Adam_1/readoptimize/Adam/beta2"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
:	í@
é
'optimize/Adam/update_embedding/Assign_1Assignembedding/Adam_1$optimize/Adam/update_embedding/mul_5"/device:CPU:0*
validate_shape(*
_output_shapes
:	í@*
use_locking( *
T0*
_class
loc:@embedding
ź
+optimize/Adam/update_embedding/ScatterAdd_1
ScatterAddembedding/Adam_1%optimize/Adam/update_embedding/Unique$optimize/Adam/update_embedding/mul_4(^optimize/Adam/update_embedding/Assign_1"/device:CPU:0*
_output_shapes
:	í@*
use_locking( *
Tindices0*
T0*
_class
loc:@embedding
ą
%optimize/Adam/update_embedding/Sqrt_1Sqrt+optimize/Adam/update_embedding/ScatterAdd_1"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
:	í@
Ő
$optimize/Adam/update_embedding/mul_6Mul&optimize/Adam/update_embedding/truediv)optimize/Adam/update_embedding/ScatterAdd"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
:	í@
ž
"optimize/Adam/update_embedding/addAdd%optimize/Adam/update_embedding/Sqrt_1optimize/Adam/epsilon"/device:CPU:0*
_output_shapes
:	í@*
T0*
_class
loc:@embedding
Ô
(optimize/Adam/update_embedding/truediv_1RealDiv$optimize/Adam/update_embedding/mul_6"optimize/Adam/update_embedding/add"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
:	í@
Ô
(optimize/Adam/update_embedding/AssignSub	AssignSub	embedding(optimize/Adam/update_embedding/truediv_1"/device:CPU:0*
_output_shapes
:	í@*
use_locking( *
T0*
_class
loc:@embedding
ă
)optimize/Adam/update_embedding/group_depsNoOp)^optimize/Adam/update_embedding/AssignSub*^optimize/Adam/update_embedding/ScatterAdd,^optimize/Adam/update_embedding/ScatterAdd_1"/device:CPU:0*
_class
loc:@embedding
ą
*optimize/Adam/update_conv/kernel/ApplyAdam	ApplyAdamconv/kernelconv/kernel/Adamconv/kernel/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/Adam/learning_rateoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon<optimize/gradients/cnn/conv/conv1d/ExpandDims_1_grad/Reshape*
T0*
_class
loc:@conv/kernel*
use_nesterov( *#
_output_shapes
:@*
use_locking( 
Ś
(optimize/Adam/update_conv/bias/ApplyAdam	ApplyAdam	conv/biasconv/bias/Adamconv/bias/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/Adam/learning_rateoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilonCoptimize/gradients/cnn/conv/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
°
)optimize/Adam/update_fc1/kernel/ApplyAdam	ApplyAdam
fc1/kernelfc1/kernel/Adamfc1/kernel/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/Adam/learning_rateoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilonCoptimize/gradients/score/fc1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@fc1/kernel*
use_nesterov( * 
_output_shapes
:

˘
'optimize/Adam/update_fc1/bias/ApplyAdam	ApplyAdamfc1/biasfc1/bias/Adamfc1/bias/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/Adam/learning_rateoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilonDoptimize/gradients/score/fc1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@fc1/bias*
use_nesterov( *
_output_shapes	
:
Ż
)optimize/Adam/update_fc2/kernel/ApplyAdam	ApplyAdam
fc2/kernelfc2/kernel/Adamfc2/kernel/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/Adam/learning_rateoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilonCoptimize/gradients/score/fc2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@fc2/kernel*
use_nesterov( *
_output_shapes
:	
Ą
'optimize/Adam/update_fc2/bias/ApplyAdam	ApplyAdamfc2/biasfc2/bias/Adamfc2/bias/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/Adam/learning_rateoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilonDoptimize/gradients/score/fc2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*
_class
loc:@fc2/bias
ˇ
optimize/Adam/mulMuloptimize/beta1_power/readoptimize/Adam/beta1)^optimize/Adam/update_conv/bias/ApplyAdam+^optimize/Adam/update_conv/kernel/ApplyAdam*^optimize/Adam/update_embedding/group_deps(^optimize/Adam/update_fc1/bias/ApplyAdam*^optimize/Adam/update_fc1/kernel/ApplyAdam(^optimize/Adam/update_fc2/bias/ApplyAdam*^optimize/Adam/update_fc2/kernel/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@conv/bias
Ż
optimize/Adam/AssignAssignoptimize/beta1_poweroptimize/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@conv/bias
š
optimize/Adam/mul_1Muloptimize/beta2_power/readoptimize/Adam/beta2)^optimize/Adam/update_conv/bias/ApplyAdam+^optimize/Adam/update_conv/kernel/ApplyAdam*^optimize/Adam/update_embedding/group_deps(^optimize/Adam/update_fc1/bias/ApplyAdam*^optimize/Adam/update_fc1/kernel/ApplyAdam(^optimize/Adam/update_fc2/bias/ApplyAdam*^optimize/Adam/update_fc2/kernel/ApplyAdam*
T0*
_class
loc:@conv/bias*
_output_shapes
: 
ł
optimize/Adam/Assign_1Assignoptimize/beta2_poweroptimize/Adam/mul_1*
use_locking( *
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes
: 
Î
optimize/Adam/NoOpNoOp^optimize/Adam/Assign^optimize/Adam/Assign_1)^optimize/Adam/update_conv/bias/ApplyAdam+^optimize/Adam/update_conv/kernel/ApplyAdam(^optimize/Adam/update_fc1/bias/ApplyAdam*^optimize/Adam/update_fc1/kernel/ApplyAdam(^optimize/Adam/update_fc2/bias/ApplyAdam*^optimize/Adam/update_fc2/kernel/ApplyAdam
W
optimize/Adam/NoOp_1NoOp*^optimize/Adam/update_embedding/group_deps"/device:CPU:0
A
optimize/AdamNoOp^optimize/Adam/NoOp^optimize/Adam/NoOp_1
[
accuracy/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

accuracy/ArgMaxArgMaxinput_yaccuracy/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
d
accuracy/EqualEqualaccuracy/ArgMaxscore/output*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
r
accuracy/CastCastaccuracy/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
X
accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
r
accuracy/MeanMeanaccuracy/Castaccuracy/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
P
lossScalarSummary	loss/tagsoptimize/Mean*
_output_shapes
: *
T0
Z
accuracy_1/tagsConst*
valueB B
accuracy_1*
dtype0*
_output_shapes
: 
\

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy/Mean*
_output_shapes
: *
T0
U
Merge/MergeSummaryMergeSummaryloss
accuracy_1*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
Ď
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*
valueřBőB	conv/biasBconv/bias/AdamBconv/bias/Adam_1Bconv/kernelBconv/kernel/AdamBconv/kernel/Adam_1B	embeddingBembedding/AdamBembedding/Adam_1Bfc1/biasBfc1/bias/AdamBfc1/bias/Adam_1B
fc1/kernelBfc1/kernel/AdamBfc1/kernel/Adam_1Bfc2/biasBfc2/bias/AdamBfc2/bias/Adam_1B
fc2/kernelBfc2/kernel/AdamBfc2/kernel/Adam_1Boptimize/beta1_powerBoptimize/beta2_power

save/SaveV2/shape_and_slicesConst*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
í
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices	conv/biasconv/bias/Adamconv/bias/Adam_1conv/kernelconv/kernel/Adamconv/kernel/Adam_1	embeddingembedding/Adamembedding/Adam_1fc1/biasfc1/bias/Adamfc1/bias/Adam_1
fc1/kernelfc1/kernel/Adamfc1/kernel/Adam_1fc2/biasfc2/bias/Adamfc2/bias/Adam_1
fc2/kernelfc2/kernel/Adamfc2/kernel/Adam_1optimize/beta1_poweroptimize/beta2_power*%
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
á
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueřBőB	conv/biasBconv/bias/AdamBconv/bias/Adam_1Bconv/kernelBconv/kernel/AdamBconv/kernel/Adam_1B	embeddingBembedding/AdamBembedding/Adam_1Bfc1/biasBfc1/bias/AdamBfc1/bias/Adam_1B
fc1/kernelBfc1/kernel/AdamBfc1/kernel/Adam_1Bfc2/biasBfc2/bias/AdamBfc2/bias/Adam_1B
fc2/kernelBfc2/kernel/AdamBfc2/kernel/Adam_1Boptimize/beta1_powerBoptimize/beta2_power*
dtype0*
_output_shapes
:
Ł
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2

save/AssignAssign	conv/biassave/RestoreV2*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv/bias
Ś
save/Assign_1Assignconv/bias/Adamsave/RestoreV2:1*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv/bias
¨
save/Assign_2Assignconv/bias/Adam_1save/RestoreV2:2*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
­
save/Assign_3Assignconv/kernelsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@conv/kernel*
validate_shape(*#
_output_shapes
:@
˛
save/Assign_4Assignconv/kernel/Adamsave/RestoreV2:4*
use_locking(*
T0*
_class
loc:@conv/kernel*
validate_shape(*#
_output_shapes
:@
´
save/Assign_5Assignconv/kernel/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@conv/kernel*
validate_shape(*#
_output_shapes
:@
´
save/Assign_6Assign	embeddingsave/RestoreV2:6"/device:CPU:0*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
:	í@
š
save/Assign_7Assignembedding/Adamsave/RestoreV2:7"/device:CPU:0*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
:	í@
ť
save/Assign_8Assignembedding/Adam_1save/RestoreV2:8"/device:CPU:0*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
:	í@

save/Assign_9Assignfc1/biassave/RestoreV2:9*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@fc1/bias
Ś
save/Assign_10Assignfc1/bias/Adamsave/RestoreV2:10*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@fc1/bias
¨
save/Assign_11Assignfc1/bias/Adam_1save/RestoreV2:11*
T0*
_class
loc:@fc1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ş
save/Assign_12Assign
fc1/kernelsave/RestoreV2:12*
use_locking(*
T0*
_class
loc:@fc1/kernel*
validate_shape(* 
_output_shapes
:

Ż
save/Assign_13Assignfc1/kernel/Adamsave/RestoreV2:13*
T0*
_class
loc:@fc1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ą
save/Assign_14Assignfc1/kernel/Adam_1save/RestoreV2:14*
T0*
_class
loc:@fc1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
 
save/Assign_15Assignfc2/biassave/RestoreV2:15*
use_locking(*
T0*
_class
loc:@fc2/bias*
validate_shape(*
_output_shapes
:
Ľ
save/Assign_16Assignfc2/bias/Adamsave/RestoreV2:16*
use_locking(*
T0*
_class
loc:@fc2/bias*
validate_shape(*
_output_shapes
:
§
save/Assign_17Assignfc2/bias/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@fc2/bias
Š
save/Assign_18Assign
fc2/kernelsave/RestoreV2:18*
use_locking(*
T0*
_class
loc:@fc2/kernel*
validate_shape(*
_output_shapes
:	
Ž
save/Assign_19Assignfc2/kernel/Adamsave/RestoreV2:19*
use_locking(*
T0*
_class
loc:@fc2/kernel*
validate_shape(*
_output_shapes
:	
°
save/Assign_20Assignfc2/kernel/Adam_1save/RestoreV2:20*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@fc2/kernel
Š
save/Assign_21Assignoptimize/beta1_powersave/RestoreV2:21*
use_locking(*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes
: 
Š
save/Assign_22Assignoptimize/beta2_powersave/RestoreV2:22*
use_locking(*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes
: 
č
save/restore_all/NoOpNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_9
^
save/restore_all/NoOp_1NoOp^save/Assign_6^save/Assign_7^save/Assign_8"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1
ń
	init/NoOpNoOp^conv/bias/Adam/Assign^conv/bias/Adam_1/Assign^conv/bias/Assign^conv/kernel/Adam/Assign^conv/kernel/Adam_1/Assign^conv/kernel/Assign^fc1/bias/Adam/Assign^fc1/bias/Adam_1/Assign^fc1/bias/Assign^fc1/kernel/Adam/Assign^fc1/kernel/Adam_1/Assign^fc1/kernel/Assign^fc2/bias/Adam/Assign^fc2/bias/Adam_1/Assign^fc2/bias/Assign^fc2/kernel/Adam/Assign^fc2/kernel/Adam_1/Assign^fc2/kernel/Assign^optimize/beta1_power/Assign^optimize/beta2_power/Assign
g
init/NoOp_1NoOp^embedding/Adam/Assign^embedding/Adam_1/Assign^embedding/Assign"/device:CPU:0
&
initNoOp
^init/NoOp^init/NoOp_1
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_583b09dea9154bceb4d01c5b76c29313/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
ł
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*Ő
valueËBČB	conv/biasBconv/bias/AdamBconv/bias/Adam_1Bconv/kernelBconv/kernel/AdamBconv/kernel/Adam_1Bfc1/biasBfc1/bias/AdamBfc1/bias/Adam_1B
fc1/kernelBfc1/kernel/AdamBfc1/kernel/Adam_1Bfc2/biasBfc2/bias/AdamBfc2/bias/Adam_1B
fc2/kernelBfc2/kernel/AdamBfc2/kernel/Adam_1Boptimize/beta1_powerBoptimize/beta2_power

save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B 
Ţ
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices	conv/biasconv/bias/Adamconv/bias/Adam_1conv/kernelconv/kernel/Adamconv/kernel/Adam_1fc1/biasfc1/bias/Adamfc1/bias/Adam_1
fc1/kernelfc1/kernel/Adamfc1/kernel/Adam_1fc2/biasfc2/bias/Adamfc2/bias/Adam_1
fc2/kernelfc2/kernel/Adamfc2/kernel/Adam_1optimize/beta1_poweroptimize/beta2_power"/device:CPU:0*"
dtypes
2
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
o
save_1/ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 

save_1/ShardedFilename_1ShardedFilenamesave_1/StringJoinsave_1/ShardedFilename_1/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 

save_1/SaveV2_1/tensor_namesConst"/device:CPU:0*@
value7B5B	embeddingBembedding/AdamBembedding/Adam_1*
dtype0*
_output_shapes
:
|
 save_1/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:
Â
save_1/SaveV2_1SaveV2save_1/ShardedFilename_1save_1/SaveV2_1/tensor_names save_1/SaveV2_1/shape_and_slices	embeddingembedding/Adamembedding/Adam_1"/device:CPU:0*
dtypes
2
°
save_1/control_dependency_1Identitysave_1/ShardedFilename_1^save_1/SaveV2_1"/device:CPU:0*
_output_shapes
: *
T0*+
_class!
loc:@save_1/ShardedFilename_1
ę
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilenamesave_1/ShardedFilename_1^save_1/control_dependency^save_1/control_dependency_1"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
Ż
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency^save_1/control_dependency_1"/device:CPU:0*
_output_shapes
: *
T0
ś
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*Ő
valueËBČB	conv/biasBconv/bias/AdamBconv/bias/Adam_1Bconv/kernelBconv/kernel/AdamBconv/kernel/Adam_1Bfc1/biasBfc1/bias/AdamBfc1/bias/Adam_1B
fc1/kernelBfc1/kernel/AdamBfc1/kernel/Adam_1Bfc2/biasBfc2/bias/AdamBfc2/bias/Adam_1B
fc2/kernelBfc2/kernel/AdamBfc2/kernel/Adam_1Boptimize/beta1_powerBoptimize/beta2_power*
dtype0*
_output_shapes
:

!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B 

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
Ą
save_1/AssignAssign	conv/biassave_1/RestoreV2*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv/bias
Ş
save_1/Assign_1Assignconv/bias/Adamsave_1/RestoreV2:1*
use_locking(*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes	
:
Ź
save_1/Assign_2Assignconv/bias/Adam_1save_1/RestoreV2:2*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv/bias
ą
save_1/Assign_3Assignconv/kernelsave_1/RestoreV2:3*
use_locking(*
T0*
_class
loc:@conv/kernel*
validate_shape(*#
_output_shapes
:@
ś
save_1/Assign_4Assignconv/kernel/Adamsave_1/RestoreV2:4*
use_locking(*
T0*
_class
loc:@conv/kernel*
validate_shape(*#
_output_shapes
:@
¸
save_1/Assign_5Assignconv/kernel/Adam_1save_1/RestoreV2:5*
use_locking(*
T0*
_class
loc:@conv/kernel*
validate_shape(*#
_output_shapes
:@
Ł
save_1/Assign_6Assignfc1/biassave_1/RestoreV2:6*
use_locking(*
T0*
_class
loc:@fc1/bias*
validate_shape(*
_output_shapes	
:
¨
save_1/Assign_7Assignfc1/bias/Adamsave_1/RestoreV2:7*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@fc1/bias
Ş
save_1/Assign_8Assignfc1/bias/Adam_1save_1/RestoreV2:8*
use_locking(*
T0*
_class
loc:@fc1/bias*
validate_shape(*
_output_shapes	
:
Ź
save_1/Assign_9Assign
fc1/kernelsave_1/RestoreV2:9*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@fc1/kernel
ł
save_1/Assign_10Assignfc1/kernel/Adamsave_1/RestoreV2:10*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@fc1/kernel
ľ
save_1/Assign_11Assignfc1/kernel/Adam_1save_1/RestoreV2:11*
T0*
_class
loc:@fc1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
¤
save_1/Assign_12Assignfc2/biassave_1/RestoreV2:12*
use_locking(*
T0*
_class
loc:@fc2/bias*
validate_shape(*
_output_shapes
:
Š
save_1/Assign_13Assignfc2/bias/Adamsave_1/RestoreV2:13*
use_locking(*
T0*
_class
loc:@fc2/bias*
validate_shape(*
_output_shapes
:
Ť
save_1/Assign_14Assignfc2/bias/Adam_1save_1/RestoreV2:14*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@fc2/bias
­
save_1/Assign_15Assign
fc2/kernelsave_1/RestoreV2:15*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@fc2/kernel
˛
save_1/Assign_16Assignfc2/kernel/Adamsave_1/RestoreV2:16*
use_locking(*
T0*
_class
loc:@fc2/kernel*
validate_shape(*
_output_shapes
:	
´
save_1/Assign_17Assignfc2/kernel/Adam_1save_1/RestoreV2:17*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@fc2/kernel
­
save_1/Assign_18Assignoptimize/beta1_powersave_1/RestoreV2:18*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes
: *
use_locking(
­
save_1/Assign_19Assignoptimize/beta2_powersave_1/RestoreV2:19*
T0*
_class
loc:@conv/bias*
validate_shape(*
_output_shapes
: *
use_locking(

save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
˘
save_1/RestoreV2_1/tensor_namesConst"/device:CPU:0*@
value7B5B	embeddingBembedding/AdamBembedding/Adam_1*
dtype0*
_output_shapes
:

#save_1/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:
ˇ
save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes
2* 
_output_shapes
:::
š
save_1/Assign_20Assign	embeddingsave_1/RestoreV2_1"/device:CPU:0*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
:	í@
Ŕ
save_1/Assign_21Assignembedding/Adamsave_1/RestoreV2_1:1"/device:CPU:0*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
:	í@*
use_locking(
Â
save_1/Assign_22Assignembedding/Adam_1save_1/RestoreV2_1:2"/device:CPU:0*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
:	í@*
use_locking(
f
save_1/restore_shard_1NoOp^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22"/device:CPU:0
6
save_1/restore_all/NoOpNoOp^save_1/restore_shard
I
save_1/restore_all/NoOp_1NoOp^save_1/restore_shard_1"/device:CPU:0
P
save_1/restore_allNoOp^save_1/restore_all/NoOp^save_1/restore_all/NoOp_1"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"%
	summaries

loss:0
accuracy_1:0"
trainable_variablesűř
[
embedding:0embedding/Assignembedding/read:02&embedding/Initializer/random_uniform:08
c
conv/kernel:0conv/kernel/Assignconv/kernel/read:02(conv/kernel/Initializer/random_uniform:08
R
conv/bias:0conv/bias/Assignconv/bias/read:02conv/bias/Initializer/zeros:08
_
fc1/kernel:0fc1/kernel/Assignfc1/kernel/read:02'fc1/kernel/Initializer/random_uniform:08
N

fc1/bias:0fc1/bias/Assignfc1/bias/read:02fc1/bias/Initializer/zeros:08
_
fc2/kernel:0fc2/kernel/Assignfc2/kernel/read:02'fc2/kernel/Initializer/random_uniform:08
N

fc2/bias:0fc2/bias/Assignfc2/bias/read:02fc2/bias/Initializer/zeros:08"
train_op

optimize/Adam"Ů
	variablesËČ
[
embedding:0embedding/Assignembedding/read:02&embedding/Initializer/random_uniform:08
c
conv/kernel:0conv/kernel/Assignconv/kernel/read:02(conv/kernel/Initializer/random_uniform:08
R
conv/bias:0conv/bias/Assignconv/bias/read:02conv/bias/Initializer/zeros:08
_
fc1/kernel:0fc1/kernel/Assignfc1/kernel/read:02'fc1/kernel/Initializer/random_uniform:08
N

fc1/bias:0fc1/bias/Assignfc1/bias/read:02fc1/bias/Initializer/zeros:08
_
fc2/kernel:0fc2/kernel/Assignfc2/kernel/read:02'fc2/kernel/Initializer/random_uniform:08
N

fc2/bias:0fc2/bias/Assignfc2/bias/read:02fc2/bias/Initializer/zeros:08
x
optimize/beta1_power:0optimize/beta1_power/Assignoptimize/beta1_power/read:02$optimize/beta1_power/initial_value:0
x
optimize/beta2_power:0optimize/beta2_power/Assignoptimize/beta2_power/read:02$optimize/beta2_power/initial_value:0
d
embedding/Adam:0embedding/Adam/Assignembedding/Adam/read:02"embedding/Adam/Initializer/zeros:0
l
embedding/Adam_1:0embedding/Adam_1/Assignembedding/Adam_1/read:02$embedding/Adam_1/Initializer/zeros:0
l
conv/kernel/Adam:0conv/kernel/Adam/Assignconv/kernel/Adam/read:02$conv/kernel/Adam/Initializer/zeros:0
t
conv/kernel/Adam_1:0conv/kernel/Adam_1/Assignconv/kernel/Adam_1/read:02&conv/kernel/Adam_1/Initializer/zeros:0
d
conv/bias/Adam:0conv/bias/Adam/Assignconv/bias/Adam/read:02"conv/bias/Adam/Initializer/zeros:0
l
conv/bias/Adam_1:0conv/bias/Adam_1/Assignconv/bias/Adam_1/read:02$conv/bias/Adam_1/Initializer/zeros:0
h
fc1/kernel/Adam:0fc1/kernel/Adam/Assignfc1/kernel/Adam/read:02#fc1/kernel/Adam/Initializer/zeros:0
p
fc1/kernel/Adam_1:0fc1/kernel/Adam_1/Assignfc1/kernel/Adam_1/read:02%fc1/kernel/Adam_1/Initializer/zeros:0
`
fc1/bias/Adam:0fc1/bias/Adam/Assignfc1/bias/Adam/read:02!fc1/bias/Adam/Initializer/zeros:0
h
fc1/bias/Adam_1:0fc1/bias/Adam_1/Assignfc1/bias/Adam_1/read:02#fc1/bias/Adam_1/Initializer/zeros:0
h
fc2/kernel/Adam:0fc2/kernel/Adam/Assignfc2/kernel/Adam/read:02#fc2/kernel/Adam/Initializer/zeros:0
p
fc2/kernel/Adam_1:0fc2/kernel/Adam_1/Assignfc2/kernel/Adam_1/read:02%fc2/kernel/Adam_1/Initializer/zeros:0
`
fc2/bias/Adam:0fc2/bias/Adam/Assignfc2/bias/Adam/read:02!fc2/bias/Adam/Initializer/zeros:0
h
fc2/bias/Adam_1:0fc2/bias/Adam_1/Assignfc2/bias/Adam_1/read:02#fc2/bias/Adam_1/Initializer/zeros:0*
inputs
 
	keep_prob
keep_prob:0
,
input_x!
	input_x:0˙˙˙˙˙˙˙˙˙Ř
+
input_y 
	input_y:0˙˙˙˙˙˙˙˙˙
loss
optimize/Mean:0 +
prob#
score/prob:0˙˙˙˙˙˙˙˙˙+
output!
score/output:0	˙˙˙˙˙˙˙˙˙
acc
accuracy/Mean:0 