вп
∙╛
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
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
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.0-dev202012142v1.12.1-47539-g9697081dac28Вз
В
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: *
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
: *
dtype0
В
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
: *
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
аА*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
аА*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:А*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	А
*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:
*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
█"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ц"
valueМ"BЙ" BВ"
┴
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
 regularization_losses
!trainable_variables
"	keras_api
R
#	variables
$regularization_losses
%trainable_variables
&	keras_api
h

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
h

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
R
3	variables
4regularization_losses
5trainable_variables
6	keras_api
 
 
8
0
1
2
3
'4
(5
-6
.7
8
0
1
2
3
'4
(5
-6
.7
н
7layer_metrics

8layers
9metrics
:layer_regularization_losses
;non_trainable_variables

regularization_losses
	variables
trainable_variables
 
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
<layer_metrics

=layers
>layer_regularization_losses
?non_trainable_variables
	variables
regularization_losses
@metrics
trainable_variables
 
 
 
н
Alayer_metrics

Blayers
Clayer_regularization_losses
Dnon_trainable_variables
	variables
regularization_losses
Emetrics
trainable_variables
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
Flayer_metrics

Glayers
Hlayer_regularization_losses
Inon_trainable_variables
	variables
regularization_losses
Jmetrics
trainable_variables
 
 
 
н
Klayer_metrics

Llayers
Mlayer_regularization_losses
Nnon_trainable_variables
	variables
 regularization_losses
Ometrics
!trainable_variables
 
 
 
н
Player_metrics

Qlayers
Rlayer_regularization_losses
Snon_trainable_variables
#	variables
$regularization_losses
Tmetrics
%trainable_variables
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
н
Ulayer_metrics

Vlayers
Wlayer_regularization_losses
Xnon_trainable_variables
)	variables
*regularization_losses
Ymetrics
+trainable_variables
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
н
Zlayer_metrics

[layers
\layer_regularization_losses
]non_trainable_variables
/	variables
0regularization_losses
^metrics
1trainable_variables
 
 
 
н
_layer_metrics

`layers
alayer_regularization_losses
bnon_trainable_variables
3	variables
4regularization_losses
cmetrics
5trainable_variables
 
8
0
1
2
3
4
5
6
7

d0
e1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ftotal
	gcount
h	variables
i	keras_api
D
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

h	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

m	variables
К
serving_default_input_4Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
╛
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_258937
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
│
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8В *(
f#R!
__inference__traced_save_259233
╛
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biastotalcounttotal_1count_1*
Tin
2*
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
GPU 2J 8В *+
f&R$
"__inference__traced_restore_259279Яс
Ф
¤
D__inference_conv2d_7_layer_call_and_return_conditional_losses_259094

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╛
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_258697

inputs
identityТ
MaxPoolMaxPoolinputs*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ф
¤
D__inference_conv2d_6_layer_call_and_return_conditional_losses_258636

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
х
a
E__inference_softmax_3_layer_call_and_return_conditional_losses_258778

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
Є#
К
H__inference_sequential_3_layer_call_and_return_conditional_losses_258815
input_4)
conv2d_6_258790: 
conv2d_6_258792: )
conv2d_7_258796:  
conv2d_7_258798: "
dense_6_258803:
аА
dense_6_258805:	А!
dense_7_258808:	А

dense_7_258810:

identityИв conv2d_6/StatefulPartitionedCallв conv2d_7/StatefulPartitionedCallвdense_6/StatefulPartitionedCallвdense_7/StatefulPartitionedCallЭ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_6_258790conv2d_6_258792*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2586362"
 conv2d_6/StatefulPartitionedCallФ
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2586572!
max_pooling2d_6/PartitionedCall╛
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_258796conv2d_7_258798*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2586762"
 conv2d_7/StatefulPartitionedCallФ
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2586972!
max_pooling2d_7/PartitionedCall·
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2587112
flatten_3/PartitionedCallм
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_258803dense_6_258805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2587302!
dense_6/StatefulPartitionedCall▒
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_258808dense_7_258810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2587572!
dense_7/StatefulPartitionedCall∙
softmax_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_softmax_3_layer_call_and_return_conditional_losses_2587782
softmax_3/PartitionedCallА
IdentityIdentity"softmax_3/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_4
├
Ю
)__inference_conv2d_7_layer_call_fn_259103

inputs!
unknown:  
	unknown_0: 
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2586762
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╦;
ї
!__inference__wrapped_model_258621
input_4N
4sequential_3_conv2d_6_conv2d_readvariableop_resource: C
5sequential_3_conv2d_6_biasadd_readvariableop_resource: N
4sequential_3_conv2d_7_conv2d_readvariableop_resource:  C
5sequential_3_conv2d_7_biasadd_readvariableop_resource: G
3sequential_3_dense_6_matmul_readvariableop_resource:
аАC
4sequential_3_dense_6_biasadd_readvariableop_resource:	АF
3sequential_3_dense_7_matmul_readvariableop_resource:	А
B
4sequential_3_dense_7_biasadd_readvariableop_resource:

identityИв,sequential_3/conv2d_6/BiasAdd/ReadVariableOpв+sequential_3/conv2d_6/Conv2D/ReadVariableOpв,sequential_3/conv2d_7/BiasAdd/ReadVariableOpв+sequential_3/conv2d_7/Conv2D/ReadVariableOpв+sequential_3/dense_6/BiasAdd/ReadVariableOpв*sequential_3/dense_6/MatMul/ReadVariableOpв+sequential_3/dense_7/BiasAdd/ReadVariableOpв*sequential_3/dense_7/MatMul/ReadVariableOp╫
+sequential_3/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_6/Conv2D/ReadVariableOpц
sequential_3/conv2d_6/Conv2DConv2Dinput_43sequential_3/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
sequential_3/conv2d_6/Conv2D╬
,sequential_3/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_6/BiasAdd/ReadVariableOpр
sequential_3/conv2d_6/BiasAddBiasAdd%sequential_3/conv2d_6/Conv2D:output:04sequential_3/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
sequential_3/conv2d_6/BiasAddв
sequential_3/conv2d_6/ReluRelu&sequential_3/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential_3/conv2d_6/Reluю
$sequential_3/max_pooling2d_6/MaxPoolMaxPool(sequential_3/conv2d_6/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_6/MaxPool╫
+sequential_3/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+sequential_3/conv2d_7/Conv2D/ReadVariableOpМ
sequential_3/conv2d_7/Conv2DConv2D-sequential_3/max_pooling2d_6/MaxPool:output:03sequential_3/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
sequential_3/conv2d_7/Conv2D╬
,sequential_3/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_7/BiasAdd/ReadVariableOpр
sequential_3/conv2d_7/BiasAddBiasAdd%sequential_3/conv2d_7/Conv2D:output:04sequential_3/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
sequential_3/conv2d_7/BiasAddв
sequential_3/conv2d_7/ReluRelu&sequential_3/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential_3/conv2d_7/Reluю
$sequential_3/max_pooling2d_7/MaxPoolMaxPool(sequential_3/conv2d_7/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_7/MaxPoolН
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_3/flatten_3/Const╘
sequential_3/flatten_3/ReshapeReshape-sequential_3/max_pooling2d_7/MaxPool:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:         а2 
sequential_3/flatten_3/Reshape╬
*sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
аА*
dtype02,
*sequential_3/dense_6/MatMul/ReadVariableOp╘
sequential_3/dense_6/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_6/MatMul╠
+sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_3/dense_6/BiasAdd/ReadVariableOp╓
sequential_3/dense_6/BiasAddBiasAdd%sequential_3/dense_6/MatMul:product:03sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_6/BiasAddШ
sequential_3/dense_6/ReluRelu%sequential_3/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_6/Relu═
*sequential_3/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_7_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02,
*sequential_3/dense_7/MatMul/ReadVariableOp╙
sequential_3/dense_7/MatMulMatMul'sequential_3/dense_6/Relu:activations:02sequential_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential_3/dense_7/MatMul╦
+sequential_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_3/dense_7/BiasAdd/ReadVariableOp╒
sequential_3/dense_7/BiasAddBiasAdd%sequential_3/dense_7/MatMul:product:03sequential_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential_3/dense_7/BiasAddЧ
sequential_3/dense_7/ReluRelu%sequential_3/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
sequential_3/dense_7/Reluж
sequential_3/softmax_3/SoftmaxSoftmax'sequential_3/dense_7/Relu:activations:0*
T0*'
_output_shapes
:         
2 
sequential_3/softmax_3/Softmaxь
IdentityIdentity(sequential_3/softmax_3/Softmax:softmax:0-^sequential_3/conv2d_6/BiasAdd/ReadVariableOp,^sequential_3/conv2d_6/Conv2D/ReadVariableOp-^sequential_3/conv2d_7/BiasAdd/ReadVariableOp,^sequential_3/conv2d_7/Conv2D/ReadVariableOp,^sequential_3/dense_6/BiasAdd/ReadVariableOp+^sequential_3/dense_6/MatMul/ReadVariableOp,^sequential_3/dense_7/BiasAdd/ReadVariableOp+^sequential_3/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2\
,sequential_3/conv2d_6/BiasAdd/ReadVariableOp,sequential_3/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_6/Conv2D/ReadVariableOp+sequential_3/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_7/BiasAdd/ReadVariableOp,sequential_3/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_7/Conv2D/ReadVariableOp+sequential_3/conv2d_7/Conv2D/ReadVariableOp2Z
+sequential_3/dense_6/BiasAdd/ReadVariableOp+sequential_3/dense_6/BiasAdd/ReadVariableOp2X
*sequential_3/dense_6/MatMul/ReadVariableOp*sequential_3/dense_6/MatMul/ReadVariableOp2Z
+sequential_3/dense_7/BiasAdd/ReadVariableOp+sequential_3/dense_7/BiasAdd/ReadVariableOp2X
*sequential_3/dense_7/MatMul/ReadVariableOp*sequential_3/dense_7/MatMul/ReadVariableOp:X T
/
_output_shapes
:         
!
_user_specified_name	input_4
я#
Й
H__inference_sequential_3_layer_call_and_return_conditional_losses_258846

inputs)
conv2d_6_258821: 
conv2d_6_258823: )
conv2d_7_258827:  
conv2d_7_258829: "
dense_6_258834:
аА
dense_6_258836:	А!
dense_7_258839:	А

dense_7_258841:

identityИв conv2d_6/StatefulPartitionedCallв conv2d_7/StatefulPartitionedCallвdense_6/StatefulPartitionedCallвdense_7/StatefulPartitionedCallЬ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_258821conv2d_6_258823*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2586362"
 conv2d_6/StatefulPartitionedCallФ
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2586572!
max_pooling2d_6/PartitionedCall╛
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_258827conv2d_7_258829*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2586762"
 conv2d_7/StatefulPartitionedCallФ
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2586972!
max_pooling2d_7/PartitionedCall·
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2587112
flatten_3/PartitionedCallм
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_258834dense_6_258836*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2587302!
dense_6/StatefulPartitionedCall▒
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_258839dense_7_258841*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2587572!
dense_7/StatefulPartitionedCall∙
softmax_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_softmax_3_layer_call_and_return_conditional_losses_2587782
softmax_3/PartitionedCallА
IdentityIdentity"softmax_3/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ъ
L
0__inference_max_pooling2d_7_layer_call_fn_259113

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2586972
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
├	
╨
-__inference_sequential_3_layer_call_fn_259032

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
аА
	unknown_4:	А
	unknown_5:	А

	unknown_6:

identityИвStatefulPartitionedCall╞
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2588462
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╛
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_259108

inputs
identityТ
MaxPoolMaxPoolinputs*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
а
Ш
(__inference_dense_6_layer_call_fn_259144

inputs
unknown:
аА
	unknown_0:	А
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2587302
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         а
 
_user_specified_nameinputs
Е$
№
__inference__traced_save_259233
file_prefix.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename▒
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*├
value╣B╢B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesв
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesж
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*s
_input_shapesb
`: : : :  : :
аА:А:	А
:
: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :&"
 
_output_shapes
:
аА:!

_output_shapes	
:А:%!

_output_shapes
:	А
: 

_output_shapes
:
:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╞	
╤
-__inference_sequential_3_layer_call_fn_258914
input_4!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
аА
	unknown_4:	А
	unknown_5:	А

	unknown_6:

identityИвStatefulPartitionedCall╟
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2588952
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_4
╛
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_258657

inputs
identityТ
MaxPoolMaxPoolinputs*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╛
F
*__inference_softmax_3_layer_call_fn_259174

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_softmax_3_layer_call_and_return_conditional_losses_2587782
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
Ф
¤
D__inference_conv2d_6_layer_call_and_return_conditional_losses_259064

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Є#
К
H__inference_sequential_3_layer_call_and_return_conditional_losses_258787
input_4)
conv2d_6_258647: 
conv2d_6_258649: )
conv2d_7_258687:  
conv2d_7_258689: "
dense_6_258741:
аА
dense_6_258743:	А!
dense_7_258768:	А

dense_7_258770:

identityИв conv2d_6/StatefulPartitionedCallв conv2d_7/StatefulPartitionedCallвdense_6/StatefulPartitionedCallвdense_7/StatefulPartitionedCallЭ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_6_258647conv2d_6_258649*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2586362"
 conv2d_6/StatefulPartitionedCallФ
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2586572!
max_pooling2d_6/PartitionedCall╛
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_258687conv2d_7_258689*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2586762"
 conv2d_7/StatefulPartitionedCallФ
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2586972!
max_pooling2d_7/PartitionedCall·
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2587112
flatten_3/PartitionedCallм
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_258741dense_6_258743*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2587302!
dense_6/StatefulPartitionedCall▒
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_258768dense_7_258770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2587572!
dense_7/StatefulPartitionedCall∙
softmax_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_softmax_3_layer_call_and_return_conditional_losses_2587782
softmax_3/PartitionedCallА
IdentityIdentity"softmax_3/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_4
Ь
Ц
(__inference_dense_7_layer_call_fn_259164

inputs
unknown:	А

	unknown_0:

identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2587572
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
п

ї
C__inference_dense_7_layer_call_and_return_conditional_losses_259155

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         
2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╧5
Б
"__inference__traced_restore_259279
file_prefix:
 assignvariableop_conv2d_6_kernel: .
 assignvariableop_1_conv2d_6_bias: <
"assignvariableop_2_conv2d_7_kernel:  .
 assignvariableop_3_conv2d_7_bias: 5
!assignvariableop_4_dense_6_kernel:
аА.
assignvariableop_5_dense_6_bias:	А4
!assignvariableop_6_dense_7_kernel:	А
-
assignvariableop_7_dense_7_bias:
"
assignvariableop_8_total: "
assignvariableop_9_count: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: 
identity_13ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╖
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*├
value╣B╢B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesи
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_conv2d_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2з
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3е
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ж
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5д
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ж
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7д
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Э
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Э
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10г
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11г
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpц
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12┘
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
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
_user_specified_namefile_prefix
╨
F
*__inference_flatten_3_layer_call_fn_259124

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2587112
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         а2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ф
¤
D__inference_conv2d_7_layer_call_and_return_conditional_losses_258676

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
х
a
E__inference_softmax_3_layer_call_and_return_conditional_losses_259169

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
├
Ю
)__inference_conv2d_6_layer_call_fn_259073

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2586362
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ц	
╚
$__inference_signature_wrapper_258937
input_4!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
аА
	unknown_4:	А
	unknown_5:	А

	unknown_6:

identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_2586212
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_4
ч
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_259119

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         а2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         а2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╛
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_259078

inputs
identityТ
MaxPoolMaxPoolinputs*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
п

ї
C__inference_dense_7_layer_call_and_return_conditional_losses_258757

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         
2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ъ
L
0__inference_max_pooling2d_6_layer_call_fn_259083

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2586572
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
├	
╨
-__inference_sequential_3_layer_call_fn_259053

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
аА
	unknown_4:	А
	unknown_5:	А

	unknown_6:

identityИвStatefulPartitionedCall╞
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2588952
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
я#
Й
H__inference_sequential_3_layer_call_and_return_conditional_losses_258895

inputs)
conv2d_6_258870: 
conv2d_6_258872: )
conv2d_7_258876:  
conv2d_7_258878: "
dense_6_258883:
аА
dense_6_258885:	А!
dense_7_258888:	А

dense_7_258890:

identityИв conv2d_6/StatefulPartitionedCallв conv2d_7/StatefulPartitionedCallвdense_6/StatefulPartitionedCallвdense_7/StatefulPartitionedCallЬ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_258870conv2d_6_258872*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2586362"
 conv2d_6/StatefulPartitionedCallФ
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2586572!
max_pooling2d_6/PartitionedCall╛
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_258876conv2d_7_258878*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2586762"
 conv2d_7/StatefulPartitionedCallФ
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2586972!
max_pooling2d_7/PartitionedCall·
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2587112
flatten_3/PartitionedCallм
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_258883dense_6_258885*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2587302!
dense_6/StatefulPartitionedCall▒
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_258888dense_7_258890*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2587572!
dense_7/StatefulPartitionedCall∙
softmax_3/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_softmax_3_layer_call_and_return_conditional_losses_2587782
softmax_3/PartitionedCallА
IdentityIdentity"softmax_3/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
к/
╦
H__inference_sequential_3_layer_call_and_return_conditional_losses_259011

inputsA
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource:  6
(conv2d_7_biasadd_readvariableop_resource: :
&dense_6_matmul_readvariableop_resource:
аА6
'dense_6_biasadd_readvariableop_resource:	А9
&dense_7_matmul_readvariableop_resource:	А
5
'dense_7_biasadd_readvariableop_resource:

identityИвconv2d_6/BiasAdd/ReadVariableOpвconv2d_6/Conv2D/ReadVariableOpвconv2d_7/BiasAdd/ReadVariableOpвconv2d_7/Conv2D/ReadVariableOpвdense_6/BiasAdd/ReadVariableOpвdense_6/MatMul/ReadVariableOpвdense_7/BiasAdd/ReadVariableOpвdense_7/MatMul/ReadVariableOp░
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp╛
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_6/Conv2Dз
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOpм
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_6/Relu╟
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool░
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_7/Conv2D/ReadVariableOp╪
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_7/Conv2Dз
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_7/BiasAdd/ReadVariableOpм
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_7/Relu╟
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_3/Constа
flatten_3/ReshapeReshape max_pooling2d_7/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:         а2
flatten_3/Reshapeз
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
аА*
dtype02
dense_6/MatMul/ReadVariableOpа
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_6/MatMulе
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_6/BiasAdd/ReadVariableOpв
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_6/Reluж
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
dense_7/MatMul/ReadVariableOpЯ
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_7/MatMulд
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_7/BiasAdd/ReadVariableOpб
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_7/Relu
softmax_3/SoftmaxSoftmaxdense_7/Relu:activations:0*
T0*'
_output_shapes
:         
2
softmax_3/Softmaxў
IdentityIdentitysoftmax_3/Softmax:softmax:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╖

ў
C__inference_dense_6_layer_call_and_return_conditional_losses_258730

inputs2
matmul_readvariableop_resource:
аА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
аА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         а
 
_user_specified_nameinputs
ч
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_258711

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         а2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         а2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╖

ў
C__inference_dense_6_layer_call_and_return_conditional_losses_259135

inputs2
matmul_readvariableop_resource:
аА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
аА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         а
 
_user_specified_nameinputs
╞	
╤
-__inference_sequential_3_layer_call_fn_258865
input_4!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
аА
	unknown_4:	А
	unknown_5:	А

	unknown_6:

identityИвStatefulPartitionedCall╟
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2588462
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_4
к/
╦
H__inference_sequential_3_layer_call_and_return_conditional_losses_258974

inputsA
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource:  6
(conv2d_7_biasadd_readvariableop_resource: :
&dense_6_matmul_readvariableop_resource:
аА6
'dense_6_biasadd_readvariableop_resource:	А9
&dense_7_matmul_readvariableop_resource:	А
5
'dense_7_biasadd_readvariableop_resource:

identityИвconv2d_6/BiasAdd/ReadVariableOpвconv2d_6/Conv2D/ReadVariableOpвconv2d_7/BiasAdd/ReadVariableOpвconv2d_7/Conv2D/ReadVariableOpвdense_6/BiasAdd/ReadVariableOpвdense_6/MatMul/ReadVariableOpвdense_7/BiasAdd/ReadVariableOpвdense_7/MatMul/ReadVariableOp░
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp╛
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_6/Conv2Dз
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOpм
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_6/Relu╟
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool░
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_7/Conv2D/ReadVariableOp╪
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_7/Conv2Dз
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_7/BiasAdd/ReadVariableOpм
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_7/Relu╟
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_3/Constа
flatten_3/ReshapeReshape max_pooling2d_7/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:         а2
flatten_3/Reshapeз
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
аА*
dtype02
dense_6/MatMul/ReadVariableOpа
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_6/MatMulе
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_6/BiasAdd/ReadVariableOpв
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_6/Reluж
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
dense_7/MatMul/ReadVariableOpЯ
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_7/MatMulд
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_7/BiasAdd/ReadVariableOpб
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_7/Relu
softmax_3/SoftmaxSoftmaxdense_7/Relu:activations:0*
T0*'
_output_shapes
:         
2
softmax_3/Softmaxў
IdentityIdentitysoftmax_3/Softmax:softmax:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs"цL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┤
serving_defaultа
C
input_48
serving_default_input_4:0         =
	softmax_30
StatefulPartitionedCall:0         
tensorflow/serving/predict:╢Є
╛>
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
o__call__
p_default_save_signature
*q&call_and_return_all_conditional_losses"г;
_tf_keras_sequentialД;{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Softmax", "config": {"name": "softmax_3", "trainable": true, "dtype": "float32", "axis": -1}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "float32", "input_4"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Softmax", "config": {"name": "softmax_3", "trainable": true, "dtype": "float32", "axis": -1}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.01, "decay": 0.0, "momentum": 0.99, "nesterov": false}}}}
Ё	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"╦
_tf_keras_layer▒{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
 
	variables
regularization_losses
trainable_variables
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"class_name": "MaxPooling2D", "name": "max_pooling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Є	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
 
	variables
 regularization_losses
!trainable_variables
"	keras_api
x__call__
*y&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"class_name": "MaxPooling2D", "name": "max_pooling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ц
#	variables
$regularization_losses
%trainable_variables
&	keras_api
z__call__
*{&call_and_return_all_conditional_losses"╫
_tf_keras_layer╜{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ї

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
|__call__
*}&call_and_return_all_conditional_losses"╨
_tf_keras_layer╢{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1568}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1568]}}
Є

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
~__call__
*&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
─
3	variables
4regularization_losses
5trainable_variables
6	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"│
_tf_keras_layerЩ{"class_name": "Softmax", "name": "softmax_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax_3", "trainable": true, "dtype": "float32", "axis": -1}}
"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
'4
(5
-6
.7"
trackable_list_wrapper
X
0
1
2
3
'4
(5
-6
.7"
trackable_list_wrapper
╩
7layer_metrics

8layers
9metrics
:layer_regularization_losses
;non_trainable_variables

regularization_losses
	variables
trainable_variables
o__call__
p_default_save_signature
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
-
Вserving_default"
signature_map
):' 2conv2d_6/kernel
: 2conv2d_6/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
<layer_metrics

=layers
>layer_regularization_losses
?non_trainable_variables
	variables
regularization_losses
@metrics
trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Alayer_metrics

Blayers
Clayer_regularization_losses
Dnon_trainable_variables
	variables
regularization_losses
Emetrics
trainable_variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_7/kernel
: 2conv2d_7/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
Flayer_metrics

Glayers
Hlayer_regularization_losses
Inon_trainable_variables
	variables
regularization_losses
Jmetrics
trainable_variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Klayer_metrics

Llayers
Mlayer_regularization_losses
Nnon_trainable_variables
	variables
 regularization_losses
Ometrics
!trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Player_metrics

Qlayers
Rlayer_regularization_losses
Snon_trainable_variables
#	variables
$regularization_losses
Tmetrics
%trainable_variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
": 
аА2dense_6/kernel
:А2dense_6/bias
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
н
Ulayer_metrics

Vlayers
Wlayer_regularization_losses
Xnon_trainable_variables
)	variables
*regularization_losses
Ymetrics
+trainable_variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
!:	А
2dense_7/kernel
:
2dense_7/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
н
Zlayer_metrics

[layers
\layer_regularization_losses
]non_trainable_variables
/	variables
0regularization_losses
^metrics
1trainable_variables
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
_layer_metrics

`layers
alayer_regularization_losses
bnon_trainable_variables
3	variables
4regularization_losses
cmetrics
5trainable_variables
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╗
	ftotal
	gcount
h	variables
i	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api"╕
_tf_keras_metricЭ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
В2 
-__inference_sequential_3_layer_call_fn_259053
-__inference_sequential_3_layer_call_fn_258914
-__inference_sequential_3_layer_call_fn_258865
-__inference_sequential_3_layer_call_fn_259032└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ч2ф
!__inference__wrapped_model_258621╛
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
input_4         
ю2ы
H__inference_sequential_3_layer_call_and_return_conditional_losses_258815
H__inference_sequential_3_layer_call_and_return_conditional_losses_258974
H__inference_sequential_3_layer_call_and_return_conditional_losses_259011
H__inference_sequential_3_layer_call_and_return_conditional_losses_258787└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╙2╨
)__inference_conv2d_6_layer_call_fn_259073в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_conv2d_6_layer_call_and_return_conditional_losses_259064в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_max_pooling2d_6_layer_call_fn_259083в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_259078в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_conv2d_7_layer_call_fn_259103в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_conv2d_7_layer_call_and_return_conditional_losses_259094в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_max_pooling2d_7_layer_call_fn_259113в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_259108в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_flatten_3_layer_call_fn_259124в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_flatten_3_layer_call_and_return_conditional_losses_259119в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_6_layer_call_fn_259144в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_6_layer_call_and_return_conditional_losses_259135в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_7_layer_call_fn_259164в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_7_layer_call_and_return_conditional_losses_259155в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
с2▐
*__inference_softmax_3_layer_call_fn_259174п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№2∙
E__inference_softmax_3_layer_call_and_return_conditional_losses_259169п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╦B╚
$__inference_signature_wrapper_258937input_4"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 а
!__inference__wrapped_model_258621{'(-.8в5
.в+
)К&
input_4         
к "5к2
0
	softmax_3#К 
	softmax_3         
┤
D__inference_conv2d_6_layer_call_and_return_conditional_losses_259064l7в4
-в*
(К%
inputs         
к "-в*
#К 
0          
Ъ М
)__inference_conv2d_6_layer_call_fn_259073_7в4
-в*
(К%
inputs         
к " К          ┤
D__inference_conv2d_7_layer_call_and_return_conditional_losses_259094l7в4
-в*
(К%
inputs          
к "-в*
#К 
0          
Ъ М
)__inference_conv2d_7_layer_call_fn_259103_7в4
-в*
(К%
inputs          
к " К          е
C__inference_dense_6_layer_call_and_return_conditional_losses_259135^'(0в-
&в#
!К
inputs         а
к "&в#
К
0         А
Ъ }
(__inference_dense_6_layer_call_fn_259144Q'(0в-
&в#
!К
inputs         а
к "К         Ад
C__inference_dense_7_layer_call_and_return_conditional_losses_259155]-.0в-
&в#
!К
inputs         А
к "%в"
К
0         

Ъ |
(__inference_dense_7_layer_call_fn_259164P-.0в-
&в#
!К
inputs         А
к "К         
к
E__inference_flatten_3_layer_call_and_return_conditional_losses_259119a7в4
-в*
(К%
inputs          
к "&в#
К
0         а
Ъ В
*__inference_flatten_3_layer_call_fn_259124T7в4
-в*
(К%
inputs          
к "К         а╖
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_259078h7в4
-в*
(К%
inputs          
к "-в*
#К 
0          
Ъ П
0__inference_max_pooling2d_6_layer_call_fn_259083[7в4
-в*
(К%
inputs          
к " К          ╖
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_259108h7в4
-в*
(К%
inputs          
к "-в*
#К 
0          
Ъ П
0__inference_max_pooling2d_7_layer_call_fn_259113[7в4
-в*
(К%
inputs          
к " К          ┐
H__inference_sequential_3_layer_call_and_return_conditional_losses_258787s'(-.@в=
6в3
)К&
input_4         
p

 
к "%в"
К
0         

Ъ ┐
H__inference_sequential_3_layer_call_and_return_conditional_losses_258815s'(-.@в=
6в3
)К&
input_4         
p 

 
к "%в"
К
0         

Ъ ╛
H__inference_sequential_3_layer_call_and_return_conditional_losses_258974r'(-.?в<
5в2
(К%
inputs         
p

 
к "%в"
К
0         

Ъ ╛
H__inference_sequential_3_layer_call_and_return_conditional_losses_259011r'(-.?в<
5в2
(К%
inputs         
p 

 
к "%в"
К
0         

Ъ Ч
-__inference_sequential_3_layer_call_fn_258865f'(-.@в=
6в3
)К&
input_4         
p

 
к "К         
Ч
-__inference_sequential_3_layer_call_fn_258914f'(-.@в=
6в3
)К&
input_4         
p 

 
к "К         
Ц
-__inference_sequential_3_layer_call_fn_259032e'(-.?в<
5в2
(К%
inputs         
p

 
к "К         
Ц
-__inference_sequential_3_layer_call_fn_259053e'(-.?в<
5в2
(К%
inputs         
p 

 
к "К         
п
$__inference_signature_wrapper_258937Ж'(-.Cв@
в 
9к6
4
input_4)К&
input_4         "5к2
0
	softmax_3#К 
	softmax_3         
е
E__inference_softmax_3_layer_call_and_return_conditional_losses_259169\3в0
)в&
 К
inputs         


 
к "%в"
К
0         

Ъ }
*__inference_softmax_3_layer_call_fn_259174O3в0
)в&
 К
inputs         


 
к "К         
