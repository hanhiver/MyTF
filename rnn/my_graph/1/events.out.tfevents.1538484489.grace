       �K"	  @B���Abrain.Event:2�5     ��� 	*�bB���A"��
n
PlaceholderPlaceholder*
shape:���������
*
dtype0*'
_output_shapes
:���������

p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
^
Placeholder_2Placeholder*
dtype0*
_output_shapes

:

*
shape
:


�
Variable/initial_valueConst*�
value�B�
"���~>\�=R$w?;YG=^)�>`�?�">{�z?��=��?��>��c?�X4?㹋>#"?��x?ϗ�>�~Y<�D�<��?�>���;��>_�>9I�>B��>�9V?��5?]0=��?��?�4�=�e<BBM?o�>�^>�4?�M?r�#=hL<?�s.?�f?��>V?�Q?�!?ĺ�>l&�>
�>V�=Pq+?m��</qr>Ѿ*?7�O?�x?Ĺ�=4E^?���>��%?��m?��?��y?ޞ?�C^?)��>�='=�:?G?9>�>�n�>Ɯ�>��t?+Tr?MN?%Fl?�-�=?w>��G?�J?֌?�}<?
��>��w?qZ1?t%V?���>��3?դ�>^|?�5?�=?	A-?��%>R�<^�?Uh?��?�u�>x��>�+%?t6?��>�_K?�\�>o�i?��3?�Ow?�:�;*
dtype0*
_output_shapes

:

|
Variable
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
Variable/AssignAssignVariableVariable/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:

i
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes

:

�
Variable_1/initial_valueConst*
_output_shapes

:
*A
value8B6
"(                                        *
dtype0
~

Variable_1
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
Variable_1/AssignAssign
Variable_1Variable_1/initial_value*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

:

o
Variable_1/readIdentity
Variable_1*
_output_shapes

:
*
T0*
_class
loc:@Variable_1
�
Variable_2/initial_valueConst*
_output_shapes

:
*A
value8B6
"(��>�<,?L@?�1?��J>.YO=U�{?�w�<��g>�ɻ>*
dtype0
~

Variable_2
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
Variable_2/AssignAssign
Variable_2Variable_2/initial_value*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:

o
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes

:

e
Variable_3/initial_valueConst*
valueB*    *
dtype0*
_output_shapes
:
v

Variable_3
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
Variable_3/AssignAssign
Variable_3Variable_3/initial_value*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:*
use_locking(
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:
�
unstackUnpackPlaceholder*	
num
*
T0*

axis*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������
^
Reshape/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
a
ReshapeReshapeunstackReshape/shape*
T0*
Tshape0*
_output_shapes

:

M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
u
concatConcatV2ReshapePlaceholder_2concat/axis*

Tidx0*
T0*
N*
_output_shapes

:

v
MatMulMatMulconcatVariable/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
L
addAddMatMulVariable_1/read*
T0*
_output_shapes

:


:
TanhTanhadd*
_output_shapes

:

*
T0
`
Reshape_1/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_1Reshape	unstack:1Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:

O
concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
r
concat_1ConcatV2	Reshape_1Tanhconcat_1/axis*
N*
_output_shapes

:
*

Tidx0*
T0
z
MatMul_1MatMulconcat_1Variable/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
P
add_1AddMatMul_1Variable_1/read*
T0*
_output_shapes

:


>
Tanh_1Tanhadd_1*
_output_shapes

:

*
T0
`
Reshape_2/shapeConst*
_output_shapes
:*
valueB"
      *
dtype0
g
	Reshape_2Reshape	unstack:2Reshape_2/shape*
Tshape0*
_output_shapes

:
*
T0
O
concat_2/axisConst*
value	B :*
dtype0*
_output_shapes
: 
t
concat_2ConcatV2	Reshape_2Tanh_1concat_2/axis*
T0*
N*
_output_shapes

:
*

Tidx0
z
MatMul_2MatMulconcat_2Variable/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
P
add_2AddMatMul_2Variable_1/read*
T0*
_output_shapes

:


>
Tanh_2Tanhadd_2*
_output_shapes

:

*
T0
`
Reshape_3/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_3Reshape	unstack:3Reshape_3/shape*
T0*
Tshape0*
_output_shapes

:

O
concat_3/axisConst*
value	B :*
dtype0*
_output_shapes
: 
t
concat_3ConcatV2	Reshape_3Tanh_2concat_3/axis*
T0*
N*
_output_shapes

:
*

Tidx0
z
MatMul_3MatMulconcat_3Variable/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
P
add_3AddMatMul_3Variable_1/read*
T0*
_output_shapes

:


>
Tanh_3Tanhadd_3*
T0*
_output_shapes

:


`
Reshape_4/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_4Reshape	unstack:4Reshape_4/shape*
T0*
Tshape0*
_output_shapes

:

O
concat_4/axisConst*
value	B :*
dtype0*
_output_shapes
: 
t
concat_4ConcatV2	Reshape_4Tanh_3concat_4/axis*
_output_shapes

:
*

Tidx0*
T0*
N
z
MatMul_4MatMulconcat_4Variable/read*
transpose_b( *
T0*
_output_shapes

:

*
transpose_a( 
P
add_4AddMatMul_4Variable_1/read*
T0*
_output_shapes

:


>
Tanh_4Tanhadd_4*
_output_shapes

:

*
T0
`
Reshape_5/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_5Reshape	unstack:5Reshape_5/shape*
T0*
Tshape0*
_output_shapes

:

O
concat_5/axisConst*
value	B :*
dtype0*
_output_shapes
: 
t
concat_5ConcatV2	Reshape_5Tanh_4concat_5/axis*
T0*
N*
_output_shapes

:
*

Tidx0
z
MatMul_5MatMulconcat_5Variable/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
P
add_5AddMatMul_5Variable_1/read*
T0*
_output_shapes

:


>
Tanh_5Tanhadd_5*
T0*
_output_shapes

:


`
Reshape_6/shapeConst*
dtype0*
_output_shapes
:*
valueB"
      
g
	Reshape_6Reshape	unstack:6Reshape_6/shape*
_output_shapes

:
*
T0*
Tshape0
O
concat_6/axisConst*
_output_shapes
: *
value	B :*
dtype0
t
concat_6ConcatV2	Reshape_6Tanh_5concat_6/axis*
N*
_output_shapes

:
*

Tidx0*
T0
z
MatMul_6MatMulconcat_6Variable/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
P
add_6AddMatMul_6Variable_1/read*
T0*
_output_shapes

:


>
Tanh_6Tanhadd_6*
T0*
_output_shapes

:


`
Reshape_7/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_7Reshape	unstack:7Reshape_7/shape*
T0*
Tshape0*
_output_shapes

:

O
concat_7/axisConst*
value	B :*
dtype0*
_output_shapes
: 
t
concat_7ConcatV2	Reshape_7Tanh_6concat_7/axis*
T0*
N*
_output_shapes

:
*

Tidx0
z
MatMul_7MatMulconcat_7Variable/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
P
add_7AddMatMul_7Variable_1/read*
T0*
_output_shapes

:


>
Tanh_7Tanhadd_7*
T0*
_output_shapes

:


`
Reshape_8/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_8Reshape	unstack:8Reshape_8/shape*
T0*
Tshape0*
_output_shapes

:

O
concat_8/axisConst*
value	B :*
dtype0*
_output_shapes
: 
t
concat_8ConcatV2	Reshape_8Tanh_7concat_8/axis*
N*
_output_shapes

:
*

Tidx0*
T0
z
MatMul_8MatMulconcat_8Variable/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
P
add_8AddMatMul_8Variable_1/read*
T0*
_output_shapes

:


>
Tanh_8Tanhadd_8*
T0*
_output_shapes

:


`
Reshape_9/shapeConst*
_output_shapes
:*
valueB"
      *
dtype0
g
	Reshape_9Reshape	unstack:9Reshape_9/shape*
T0*
Tshape0*
_output_shapes

:

O
concat_9/axisConst*
dtype0*
_output_shapes
: *
value	B :
t
concat_9ConcatV2	Reshape_9Tanh_8concat_9/axis*
T0*
N*
_output_shapes

:
*

Tidx0
z
MatMul_9MatMulconcat_9Variable/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
P
add_9AddMatMul_9Variable_1/read*
_output_shapes

:

*
T0
>
Tanh_9Tanhadd_9*
T0*
_output_shapes

:


{
	MatMul_10MatMulTanh_9Variable_2/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b( 
R
add_10Add	MatMul_10Variable_3/read*
T0*
_output_shapes

:

J
SubSubPlaceholder_1add_10*
T0*
_output_shapes

:

>
SquareSquareSub*
_output_shapes

:
*
T0
`
gradients/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
w
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes

:
*
T0*

index_type0
q
gradients/Square_grad/ConstConst^gradients/Fill*
valueB
 *   @*
dtype0*
_output_shapes
: 
k
gradients/Square_grad/MulMulSubgradients/Square_grad/Const*
T0*
_output_shapes

:

v
gradients/Square_grad/Mul_1Mulgradients/Fillgradients/Square_grad/Mul*
T0*
_output_shapes

:

e
gradients/Sub_grad/ShapeShapePlaceholder_1*
out_type0*
_output_shapes
:*
T0
k
gradients/Sub_grad/Shape_1Const*
valueB"
      *
dtype0*
_output_shapes
:
�
(gradients/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Sub_grad/Shapegradients/Sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/Sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/Sub_grad/ReshapeReshapegradients/Sub_grad/Sumgradients/Sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/Sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/Sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Z
gradients/Sub_grad/NegNeggradients/Sub_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/Sub_grad/Reshape_1Reshapegradients/Sub_grad/Neggradients/Sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

g
#gradients/Sub_grad/tuple/group_depsNoOp^gradients/Sub_grad/Reshape^gradients/Sub_grad/Reshape_1
�
+gradients/Sub_grad/tuple/control_dependencyIdentitygradients/Sub_grad/Reshape$^gradients/Sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/Sub_grad/Reshape*'
_output_shapes
:���������*
T0
�
-gradients/Sub_grad/tuple/control_dependency_1Identitygradients/Sub_grad/Reshape_1$^gradients/Sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Sub_grad/Reshape_1*
_output_shapes

:

l
gradients/add_10_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
gradients/add_10_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
+gradients/add_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_10_grad/Shapegradients/add_10_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_10_grad/SumSum-gradients/Sub_grad/tuple/control_dependency_1+gradients/add_10_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
gradients/add_10_grad/ReshapeReshapegradients/add_10_grad/Sumgradients/add_10_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
gradients/add_10_grad/Sum_1Sum-gradients/Sub_grad/tuple/control_dependency_1-gradients/add_10_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
gradients/add_10_grad/Reshape_1Reshapegradients/add_10_grad/Sum_1gradients/add_10_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
p
&gradients/add_10_grad/tuple/group_depsNoOp^gradients/add_10_grad/Reshape ^gradients/add_10_grad/Reshape_1
�
.gradients/add_10_grad/tuple/control_dependencyIdentitygradients/add_10_grad/Reshape'^gradients/add_10_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_10_grad/Reshape*
_output_shapes

:

�
0gradients/add_10_grad/tuple/control_dependency_1Identitygradients/add_10_grad/Reshape_1'^gradients/add_10_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_10_grad/Reshape_1*
_output_shapes
:
�
gradients/MatMul_10_grad/MatMulMatMul.gradients/add_10_grad/tuple/control_dependencyVariable_2/read*
transpose_b(*
T0*
_output_shapes

:

*
transpose_a( 
�
!gradients/MatMul_10_grad/MatMul_1MatMulTanh_9.gradients/add_10_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
w
)gradients/MatMul_10_grad/tuple/group_depsNoOp ^gradients/MatMul_10_grad/MatMul"^gradients/MatMul_10_grad/MatMul_1
�
1gradients/MatMul_10_grad/tuple/control_dependencyIdentitygradients/MatMul_10_grad/MatMul*^gradients/MatMul_10_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients/MatMul_10_grad/MatMul
�
3gradients/MatMul_10_grad/tuple/control_dependency_1Identity!gradients/MatMul_10_grad/MatMul_1*^gradients/MatMul_10_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/MatMul_10_grad/MatMul_1*
_output_shapes

:

�
gradients/Tanh_9_grad/TanhGradTanhGradTanh_91gradients/MatMul_10_grad/tuple/control_dependency*
T0*
_output_shapes

:


k
gradients/add_9_grad/ShapeConst*
_output_shapes
:*
valueB"
   
   *
dtype0
m
gradients/add_9_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
�
*gradients/add_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_9_grad/Shapegradients/add_9_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_9_grad/SumSumgradients/Tanh_9_grad/TanhGrad*gradients/add_9_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients/add_9_grad/ReshapeReshapegradients/add_9_grad/Sumgradients/add_9_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_9_grad/Sum_1Sumgradients/Tanh_9_grad/TanhGrad,gradients/add_9_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
gradients/add_9_grad/Reshape_1Reshapegradients/add_9_grad/Sum_1gradients/add_9_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

m
%gradients/add_9_grad/tuple/group_depsNoOp^gradients/add_9_grad/Reshape^gradients/add_9_grad/Reshape_1
�
-gradients/add_9_grad/tuple/control_dependencyIdentitygradients/add_9_grad/Reshape&^gradients/add_9_grad/tuple/group_deps*
_output_shapes

:

*
T0*/
_class%
#!loc:@gradients/add_9_grad/Reshape
�
/gradients/add_9_grad/tuple/control_dependency_1Identitygradients/add_9_grad/Reshape_1&^gradients/add_9_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_9_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_9_grad/MatMulMatMul-gradients/add_9_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_9_grad/MatMul_1MatMulconcat_9-gradients/add_9_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_9_grad/tuple/group_depsNoOp^gradients/MatMul_9_grad/MatMul!^gradients/MatMul_9_grad/MatMul_1
�
0gradients/MatMul_9_grad/tuple/control_dependencyIdentitygradients/MatMul_9_grad/MatMul)^gradients/MatMul_9_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_9_grad/MatMul*
_output_shapes

:

�
2gradients/MatMul_9_grad/tuple/control_dependency_1Identity gradients/MatMul_9_grad/MatMul_1)^gradients/MatMul_9_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_9_grad/MatMul_1*
_output_shapes

:

^
gradients/concat_9_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_9_grad/modFloorModconcat_9/axisgradients/concat_9_grad/Rank*
T0*
_output_shapes
: 
n
gradients/concat_9_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
p
gradients/concat_9_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_9_grad/ConcatOffsetConcatOffsetgradients/concat_9_grad/modgradients/concat_9_grad/Shapegradients/concat_9_grad/Shape_1*
N* 
_output_shapes
::
�
gradients/concat_9_grad/SliceSlice0gradients/MatMul_9_grad/tuple/control_dependency$gradients/concat_9_grad/ConcatOffsetgradients/concat_9_grad/Shape*
_output_shapes

:
*
Index0*
T0
�
gradients/concat_9_grad/Slice_1Slice0gradients/MatMul_9_grad/tuple/control_dependency&gradients/concat_9_grad/ConcatOffset:1gradients/concat_9_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
r
(gradients/concat_9_grad/tuple/group_depsNoOp^gradients/concat_9_grad/Slice ^gradients/concat_9_grad/Slice_1
�
0gradients/concat_9_grad/tuple/control_dependencyIdentitygradients/concat_9_grad/Slice)^gradients/concat_9_grad/tuple/group_deps*
_output_shapes

:
*
T0*0
_class&
$"loc:@gradients/concat_9_grad/Slice
�
2gradients/concat_9_grad/tuple/control_dependency_1Identitygradients/concat_9_grad/Slice_1)^gradients/concat_9_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_9_grad/Slice_1*
_output_shapes

:


�
gradients/Tanh_8_grad/TanhGradTanhGradTanh_82gradients/concat_9_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


k
gradients/add_8_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
m
gradients/add_8_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_8_grad/Shapegradients/add_8_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_8_grad/SumSumgradients/Tanh_8_grad/TanhGrad*gradients/add_8_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients/add_8_grad/ReshapeReshapegradients/add_8_grad/Sumgradients/add_8_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_8_grad/Sum_1Sumgradients/Tanh_8_grad/TanhGrad,gradients/add_8_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients/add_8_grad/Reshape_1Reshapegradients/add_8_grad/Sum_1gradients/add_8_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
m
%gradients/add_8_grad/tuple/group_depsNoOp^gradients/add_8_grad/Reshape^gradients/add_8_grad/Reshape_1
�
-gradients/add_8_grad/tuple/control_dependencyIdentitygradients/add_8_grad/Reshape&^gradients/add_8_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_8_grad/Reshape*
_output_shapes

:


�
/gradients/add_8_grad/tuple/control_dependency_1Identitygradients/add_8_grad/Reshape_1&^gradients/add_8_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_8_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_8_grad/MatMulMatMul-gradients/add_8_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_8_grad/MatMul_1MatMulconcat_8-gradients/add_8_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_8_grad/tuple/group_depsNoOp^gradients/MatMul_8_grad/MatMul!^gradients/MatMul_8_grad/MatMul_1
�
0gradients/MatMul_8_grad/tuple/control_dependencyIdentitygradients/MatMul_8_grad/MatMul)^gradients/MatMul_8_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/MatMul_8_grad/MatMul
�
2gradients/MatMul_8_grad/tuple/control_dependency_1Identity gradients/MatMul_8_grad/MatMul_1)^gradients/MatMul_8_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients/MatMul_8_grad/MatMul_1
^
gradients/concat_8_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
u
gradients/concat_8_grad/modFloorModconcat_8/axisgradients/concat_8_grad/Rank*
T0*
_output_shapes
: 
n
gradients/concat_8_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
p
gradients/concat_8_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_8_grad/ConcatOffsetConcatOffsetgradients/concat_8_grad/modgradients/concat_8_grad/Shapegradients/concat_8_grad/Shape_1*
N* 
_output_shapes
::
�
gradients/concat_8_grad/SliceSlice0gradients/MatMul_8_grad/tuple/control_dependency$gradients/concat_8_grad/ConcatOffsetgradients/concat_8_grad/Shape*
Index0*
T0*
_output_shapes

:

�
gradients/concat_8_grad/Slice_1Slice0gradients/MatMul_8_grad/tuple/control_dependency&gradients/concat_8_grad/ConcatOffset:1gradients/concat_8_grad/Shape_1*
Index0*
T0*
_output_shapes

:


r
(gradients/concat_8_grad/tuple/group_depsNoOp^gradients/concat_8_grad/Slice ^gradients/concat_8_grad/Slice_1
�
0gradients/concat_8_grad/tuple/control_dependencyIdentitygradients/concat_8_grad/Slice)^gradients/concat_8_grad/tuple/group_deps*
_output_shapes

:
*
T0*0
_class&
$"loc:@gradients/concat_8_grad/Slice
�
2gradients/concat_8_grad/tuple/control_dependency_1Identitygradients/concat_8_grad/Slice_1)^gradients/concat_8_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_8_grad/Slice_1*
_output_shapes

:


�
gradients/Tanh_7_grad/TanhGradTanhGradTanh_72gradients/concat_8_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


k
gradients/add_7_grad/ShapeConst*
_output_shapes
:*
valueB"
   
   *
dtype0
m
gradients/add_7_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_7_grad/Shapegradients/add_7_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_7_grad/SumSumgradients/Tanh_7_grad/TanhGrad*gradients/add_7_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients/add_7_grad/ReshapeReshapegradients/add_7_grad/Sumgradients/add_7_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_7_grad/Sum_1Sumgradients/Tanh_7_grad/TanhGrad,gradients/add_7_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
gradients/add_7_grad/Reshape_1Reshapegradients/add_7_grad/Sum_1gradients/add_7_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0
m
%gradients/add_7_grad/tuple/group_depsNoOp^gradients/add_7_grad/Reshape^gradients/add_7_grad/Reshape_1
�
-gradients/add_7_grad/tuple/control_dependencyIdentitygradients/add_7_grad/Reshape&^gradients/add_7_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_7_grad/Reshape*
_output_shapes

:


�
/gradients/add_7_grad/tuple/control_dependency_1Identitygradients/add_7_grad/Reshape_1&^gradients/add_7_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/add_7_grad/Reshape_1
�
gradients/MatMul_7_grad/MatMulMatMul-gradients/add_7_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_7_grad/MatMul_1MatMulconcat_7-gradients/add_7_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_7_grad/tuple/group_depsNoOp^gradients/MatMul_7_grad/MatMul!^gradients/MatMul_7_grad/MatMul_1
�
0gradients/MatMul_7_grad/tuple/control_dependencyIdentitygradients/MatMul_7_grad/MatMul)^gradients/MatMul_7_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/MatMul_7_grad/MatMul
�
2gradients/MatMul_7_grad/tuple/control_dependency_1Identity gradients/MatMul_7_grad/MatMul_1)^gradients/MatMul_7_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_7_grad/MatMul_1*
_output_shapes

:
*
T0
^
gradients/concat_7_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
u
gradients/concat_7_grad/modFloorModconcat_7/axisgradients/concat_7_grad/Rank*
_output_shapes
: *
T0
n
gradients/concat_7_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
p
gradients/concat_7_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_7_grad/ConcatOffsetConcatOffsetgradients/concat_7_grad/modgradients/concat_7_grad/Shapegradients/concat_7_grad/Shape_1*
N* 
_output_shapes
::
�
gradients/concat_7_grad/SliceSlice0gradients/MatMul_7_grad/tuple/control_dependency$gradients/concat_7_grad/ConcatOffsetgradients/concat_7_grad/Shape*
_output_shapes

:
*
Index0*
T0
�
gradients/concat_7_grad/Slice_1Slice0gradients/MatMul_7_grad/tuple/control_dependency&gradients/concat_7_grad/ConcatOffset:1gradients/concat_7_grad/Shape_1*
Index0*
T0*
_output_shapes

:


r
(gradients/concat_7_grad/tuple/group_depsNoOp^gradients/concat_7_grad/Slice ^gradients/concat_7_grad/Slice_1
�
0gradients/concat_7_grad/tuple/control_dependencyIdentitygradients/concat_7_grad/Slice)^gradients/concat_7_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_7_grad/Slice*
_output_shapes

:

�
2gradients/concat_7_grad/tuple/control_dependency_1Identitygradients/concat_7_grad/Slice_1)^gradients/concat_7_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients/concat_7_grad/Slice_1
�
gradients/Tanh_6_grad/TanhGradTanhGradTanh_62gradients/concat_7_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
k
gradients/add_6_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
m
gradients/add_6_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_6_grad/Shapegradients/add_6_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_6_grad/SumSumgradients/Tanh_6_grad/TanhGrad*gradients/add_6_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients/add_6_grad/ReshapeReshapegradients/add_6_grad/Sumgradients/add_6_grad/Shape*
Tshape0*
_output_shapes

:

*
T0
�
gradients/add_6_grad/Sum_1Sumgradients/Tanh_6_grad/TanhGrad,gradients/add_6_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
gradients/add_6_grad/Reshape_1Reshapegradients/add_6_grad/Sum_1gradients/add_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

m
%gradients/add_6_grad/tuple/group_depsNoOp^gradients/add_6_grad/Reshape^gradients/add_6_grad/Reshape_1
�
-gradients/add_6_grad/tuple/control_dependencyIdentitygradients/add_6_grad/Reshape&^gradients/add_6_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_6_grad/Reshape*
_output_shapes

:


�
/gradients/add_6_grad/tuple/control_dependency_1Identitygradients/add_6_grad/Reshape_1&^gradients/add_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_6_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_6_grad/MatMulMatMul-gradients/add_6_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_6_grad/MatMul_1MatMulconcat_6-gradients/add_6_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_6_grad/tuple/group_depsNoOp^gradients/MatMul_6_grad/MatMul!^gradients/MatMul_6_grad/MatMul_1
�
0gradients/MatMul_6_grad/tuple/control_dependencyIdentitygradients/MatMul_6_grad/MatMul)^gradients/MatMul_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_6_grad/MatMul*
_output_shapes

:

�
2gradients/MatMul_6_grad/tuple/control_dependency_1Identity gradients/MatMul_6_grad/MatMul_1)^gradients/MatMul_6_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_6_grad/MatMul_1*
_output_shapes

:

^
gradients/concat_6_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
u
gradients/concat_6_grad/modFloorModconcat_6/axisgradients/concat_6_grad/Rank*
_output_shapes
: *
T0
n
gradients/concat_6_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
p
gradients/concat_6_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_6_grad/ConcatOffsetConcatOffsetgradients/concat_6_grad/modgradients/concat_6_grad/Shapegradients/concat_6_grad/Shape_1*
N* 
_output_shapes
::
�
gradients/concat_6_grad/SliceSlice0gradients/MatMul_6_grad/tuple/control_dependency$gradients/concat_6_grad/ConcatOffsetgradients/concat_6_grad/Shape*
Index0*
T0*
_output_shapes

:

�
gradients/concat_6_grad/Slice_1Slice0gradients/MatMul_6_grad/tuple/control_dependency&gradients/concat_6_grad/ConcatOffset:1gradients/concat_6_grad/Shape_1*
Index0*
T0*
_output_shapes

:


r
(gradients/concat_6_grad/tuple/group_depsNoOp^gradients/concat_6_grad/Slice ^gradients/concat_6_grad/Slice_1
�
0gradients/concat_6_grad/tuple/control_dependencyIdentitygradients/concat_6_grad/Slice)^gradients/concat_6_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_6_grad/Slice*
_output_shapes

:

�
2gradients/concat_6_grad/tuple/control_dependency_1Identitygradients/concat_6_grad/Slice_1)^gradients/concat_6_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_6_grad/Slice_1*
_output_shapes

:


�
gradients/Tanh_5_grad/TanhGradTanhGradTanh_52gradients/concat_6_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
k
gradients/add_5_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
m
gradients/add_5_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_5_grad/Shapegradients/add_5_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_5_grad/SumSumgradients/Tanh_5_grad/TanhGrad*gradients/add_5_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients/add_5_grad/ReshapeReshapegradients/add_5_grad/Sumgradients/add_5_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_5_grad/Sum_1Sumgradients/Tanh_5_grad/TanhGrad,gradients/add_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
gradients/add_5_grad/Reshape_1Reshapegradients/add_5_grad/Sum_1gradients/add_5_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
m
%gradients/add_5_grad/tuple/group_depsNoOp^gradients/add_5_grad/Reshape^gradients/add_5_grad/Reshape_1
�
-gradients/add_5_grad/tuple/control_dependencyIdentitygradients/add_5_grad/Reshape&^gradients/add_5_grad/tuple/group_deps*
_output_shapes

:

*
T0*/
_class%
#!loc:@gradients/add_5_grad/Reshape
�
/gradients/add_5_grad/tuple/control_dependency_1Identitygradients/add_5_grad/Reshape_1&^gradients/add_5_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/add_5_grad/Reshape_1
�
gradients/MatMul_5_grad/MatMulMatMul-gradients/add_5_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_5_grad/MatMul_1MatMulconcat_5-gradients/add_5_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_5_grad/tuple/group_depsNoOp^gradients/MatMul_5_grad/MatMul!^gradients/MatMul_5_grad/MatMul_1
�
0gradients/MatMul_5_grad/tuple/control_dependencyIdentitygradients/MatMul_5_grad/MatMul)^gradients/MatMul_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_5_grad/MatMul*
_output_shapes

:

�
2gradients/MatMul_5_grad/tuple/control_dependency_1Identity gradients/MatMul_5_grad/MatMul_1)^gradients/MatMul_5_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_5_grad/MatMul_1*
_output_shapes

:

^
gradients/concat_5_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_5_grad/modFloorModconcat_5/axisgradients/concat_5_grad/Rank*
_output_shapes
: *
T0
n
gradients/concat_5_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
p
gradients/concat_5_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_5_grad/ConcatOffsetConcatOffsetgradients/concat_5_grad/modgradients/concat_5_grad/Shapegradients/concat_5_grad/Shape_1*
N* 
_output_shapes
::
�
gradients/concat_5_grad/SliceSlice0gradients/MatMul_5_grad/tuple/control_dependency$gradients/concat_5_grad/ConcatOffsetgradients/concat_5_grad/Shape*
Index0*
T0*
_output_shapes

:

�
gradients/concat_5_grad/Slice_1Slice0gradients/MatMul_5_grad/tuple/control_dependency&gradients/concat_5_grad/ConcatOffset:1gradients/concat_5_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
r
(gradients/concat_5_grad/tuple/group_depsNoOp^gradients/concat_5_grad/Slice ^gradients/concat_5_grad/Slice_1
�
0gradients/concat_5_grad/tuple/control_dependencyIdentitygradients/concat_5_grad/Slice)^gradients/concat_5_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_5_grad/Slice*
_output_shapes

:

�
2gradients/concat_5_grad/tuple/control_dependency_1Identitygradients/concat_5_grad/Slice_1)^gradients/concat_5_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_5_grad/Slice_1*
_output_shapes

:


�
gradients/Tanh_4_grad/TanhGradTanhGradTanh_42gradients/concat_5_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


k
gradients/add_4_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
m
gradients/add_4_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_4_grad/SumSumgradients/Tanh_4_grad/TanhGrad*gradients/add_4_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_4_grad/Sum_1Sumgradients/Tanh_4_grad/TanhGrad,gradients/add_4_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
�
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_4_grad/Reshape*
_output_shapes

:

*
T0
�
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_4_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyVariable/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_4_grad/MatMul_1MatMulconcat_4-gradients/add_4_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_4_grad/tuple/group_depsNoOp^gradients/MatMul_4_grad/MatMul!^gradients/MatMul_4_grad/MatMul_1
�
0gradients/MatMul_4_grad/tuple/control_dependencyIdentitygradients/MatMul_4_grad/MatMul)^gradients/MatMul_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_4_grad/MatMul*
_output_shapes

:

�
2gradients/MatMul_4_grad/tuple/control_dependency_1Identity gradients/MatMul_4_grad/MatMul_1)^gradients/MatMul_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_4_grad/MatMul_1*
_output_shapes

:

^
gradients/concat_4_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_4_grad/modFloorModconcat_4/axisgradients/concat_4_grad/Rank*
T0*
_output_shapes
: 
n
gradients/concat_4_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
p
gradients/concat_4_grad/Shape_1Const*
_output_shapes
:*
valueB"
   
   *
dtype0
�
$gradients/concat_4_grad/ConcatOffsetConcatOffsetgradients/concat_4_grad/modgradients/concat_4_grad/Shapegradients/concat_4_grad/Shape_1* 
_output_shapes
::*
N
�
gradients/concat_4_grad/SliceSlice0gradients/MatMul_4_grad/tuple/control_dependency$gradients/concat_4_grad/ConcatOffsetgradients/concat_4_grad/Shape*
Index0*
T0*
_output_shapes

:

�
gradients/concat_4_grad/Slice_1Slice0gradients/MatMul_4_grad/tuple/control_dependency&gradients/concat_4_grad/ConcatOffset:1gradients/concat_4_grad/Shape_1*
Index0*
T0*
_output_shapes

:


r
(gradients/concat_4_grad/tuple/group_depsNoOp^gradients/concat_4_grad/Slice ^gradients/concat_4_grad/Slice_1
�
0gradients/concat_4_grad/tuple/control_dependencyIdentitygradients/concat_4_grad/Slice)^gradients/concat_4_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_4_grad/Slice*
_output_shapes

:

�
2gradients/concat_4_grad/tuple/control_dependency_1Identitygradients/concat_4_grad/Slice_1)^gradients/concat_4_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients/concat_4_grad/Slice_1
�
gradients/Tanh_3_grad/TanhGradTanhGradTanh_32gradients/concat_4_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


k
gradients/add_3_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
m
gradients/add_3_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_3_grad/SumSumgradients/Tanh_3_grad/TanhGrad*gradients/add_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_3_grad/Sum_1Sumgradients/Tanh_3_grad/TanhGrad,gradients/add_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
_output_shapes

:


�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_3_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_3_grad/MatMul_1MatMulconcat_3-gradients/add_3_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
�
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*
_output_shapes

:

�
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1*
_output_shapes

:

^
gradients/concat_3_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_3_grad/modFloorModconcat_3/axisgradients/concat_3_grad/Rank*
_output_shapes
: *
T0
n
gradients/concat_3_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
p
gradients/concat_3_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_3_grad/ConcatOffsetConcatOffsetgradients/concat_3_grad/modgradients/concat_3_grad/Shapegradients/concat_3_grad/Shape_1*
N* 
_output_shapes
::
�
gradients/concat_3_grad/SliceSlice0gradients/MatMul_3_grad/tuple/control_dependency$gradients/concat_3_grad/ConcatOffsetgradients/concat_3_grad/Shape*
Index0*
T0*
_output_shapes

:

�
gradients/concat_3_grad/Slice_1Slice0gradients/MatMul_3_grad/tuple/control_dependency&gradients/concat_3_grad/ConcatOffset:1gradients/concat_3_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
r
(gradients/concat_3_grad/tuple/group_depsNoOp^gradients/concat_3_grad/Slice ^gradients/concat_3_grad/Slice_1
�
0gradients/concat_3_grad/tuple/control_dependencyIdentitygradients/concat_3_grad/Slice)^gradients/concat_3_grad/tuple/group_deps*
_output_shapes

:
*
T0*0
_class&
$"loc:@gradients/concat_3_grad/Slice
�
2gradients/concat_3_grad/tuple/control_dependency_1Identitygradients/concat_3_grad/Slice_1)^gradients/concat_3_grad/tuple/group_deps*2
_class(
&$loc:@gradients/concat_3_grad/Slice_1*
_output_shapes

:

*
T0
�
gradients/Tanh_2_grad/TanhGradTanhGradTanh_22gradients/concat_3_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


k
gradients/add_2_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
m
gradients/add_2_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_2_grad/SumSumgradients/Tanh_2_grad/TanhGrad*gradients/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
_output_shapes

:

*
T0*
Tshape0
�
gradients/add_2_grad/Sum_1Sumgradients/Tanh_2_grad/TanhGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
_output_shapes

:

*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_2_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_2_grad/MatMul_1MatMulconcat_2-gradients/add_2_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
�
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*
_output_shapes

:

�
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1*
_output_shapes

:

^
gradients/concat_2_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
u
gradients/concat_2_grad/modFloorModconcat_2/axisgradients/concat_2_grad/Rank*
T0*
_output_shapes
: 
n
gradients/concat_2_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
p
gradients/concat_2_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_2_grad/ConcatOffsetConcatOffsetgradients/concat_2_grad/modgradients/concat_2_grad/Shapegradients/concat_2_grad/Shape_1*
N* 
_output_shapes
::
�
gradients/concat_2_grad/SliceSlice0gradients/MatMul_2_grad/tuple/control_dependency$gradients/concat_2_grad/ConcatOffsetgradients/concat_2_grad/Shape*
Index0*
T0*
_output_shapes

:

�
gradients/concat_2_grad/Slice_1Slice0gradients/MatMul_2_grad/tuple/control_dependency&gradients/concat_2_grad/ConcatOffset:1gradients/concat_2_grad/Shape_1*
Index0*
T0*
_output_shapes

:


r
(gradients/concat_2_grad/tuple/group_depsNoOp^gradients/concat_2_grad/Slice ^gradients/concat_2_grad/Slice_1
�
0gradients/concat_2_grad/tuple/control_dependencyIdentitygradients/concat_2_grad/Slice)^gradients/concat_2_grad/tuple/group_deps*
_output_shapes

:
*
T0*0
_class&
$"loc:@gradients/concat_2_grad/Slice
�
2gradients/concat_2_grad/tuple/control_dependency_1Identitygradients/concat_2_grad/Slice_1)^gradients/concat_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_2_grad/Slice_1*
_output_shapes

:


�
gradients/Tanh_1_grad/TanhGradTanhGradTanh_12gradients/concat_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


k
gradients/add_1_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
m
gradients/add_1_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSumgradients/Tanh_1_grad/TanhGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
_output_shapes

:

*
T0
�
gradients/add_1_grad/Sum_1Sumgradients/Tanh_1_grad/TanhGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
_output_shapes

:


�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_1_grad/MatMul_1MatMulconcat_1-gradients/add_1_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
_output_shapes

:

�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_1_grad/modFloorModconcat_1/axisgradients/concat_1_grad/Rank*
T0*
_output_shapes
: 
n
gradients/concat_1_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
p
gradients/concat_1_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/modgradients/concat_1_grad/Shapegradients/concat_1_grad/Shape_1* 
_output_shapes
::*
N
�
gradients/concat_1_grad/SliceSlice0gradients/MatMul_1_grad/tuple/control_dependency$gradients/concat_1_grad/ConcatOffsetgradients/concat_1_grad/Shape*
Index0*
T0*
_output_shapes

:

�
gradients/concat_1_grad/Slice_1Slice0gradients/MatMul_1_grad/tuple/control_dependency&gradients/concat_1_grad/ConcatOffset:1gradients/concat_1_grad/Shape_1*
Index0*
T0*
_output_shapes

:


r
(gradients/concat_1_grad/tuple/group_depsNoOp^gradients/concat_1_grad/Slice ^gradients/concat_1_grad/Slice_1
�
0gradients/concat_1_grad/tuple/control_dependencyIdentitygradients/concat_1_grad/Slice)^gradients/concat_1_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_1_grad/Slice*
_output_shapes

:

�
2gradients/concat_1_grad/tuple/control_dependency_1Identitygradients/concat_1_grad/Slice_1)^gradients/concat_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_1_grad/Slice_1*
_output_shapes

:


�
gradients/Tanh_grad/TanhGradTanhGradTanh2gradients/concat_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


i
gradients/add_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
k
gradients/add_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Tanh_grad/TanhGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_grad/Sum_1Sumgradients/Tanh_grad/TanhGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*
_output_shapes

:


�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMulconcat+gradients/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes

:
*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:

�
gradients/AddNAddN/gradients/add_9_grad/tuple/control_dependency_1/gradients/add_8_grad/tuple/control_dependency_1/gradients/add_7_grad/tuple/control_dependency_1/gradients/add_6_grad/tuple/control_dependency_1/gradients/add_5_grad/tuple/control_dependency_1/gradients/add_4_grad/tuple/control_dependency_1/gradients/add_3_grad/tuple/control_dependency_1/gradients/add_2_grad/tuple/control_dependency_1/gradients/add_1_grad/tuple/control_dependency_1-gradients/add_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@gradients/add_9_grad/Reshape_1*
N
*
_output_shapes

:

�
gradients/AddN_1AddN2gradients/MatMul_9_grad/tuple/control_dependency_12gradients/MatMul_8_grad/tuple/control_dependency_12gradients/MatMul_7_grad/tuple/control_dependency_12gradients/MatMul_6_grad/tuple/control_dependency_12gradients/MatMul_5_grad/tuple/control_dependency_12gradients/MatMul_4_grad/tuple/control_dependency_12gradients/MatMul_3_grad/tuple/control_dependency_12gradients/MatMul_2_grad/tuple/control_dependency_12gradients/MatMul_1_grad/tuple/control_dependency_10gradients/MatMul_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@gradients/MatMul_9_grad/MatMul_1*
N
*
_output_shapes

:

{
beta1_power/initial_valueConst*
_class
loc:@Variable*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@Variable*
valueB
 *w�?
�
beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable*
	container 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*
_class
loc:@Variable
�
Variable/Adam/Initializer/zerosConst*
_output_shapes

:
*
_class
loc:@Variable*
valueB
*    *
dtype0
�
Variable/Adam
VariableV2*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:

s
Variable/Adam/readIdentityVariable/Adam*
_output_shapes

:
*
T0*
_class
loc:@Variable
�
!Variable/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*
_class
loc:@Variable*
valueB
*    
�
Variable/Adam_1
VariableV2*
_output_shapes

:
*
shared_name *
_class
loc:@Variable*
	container *
shape
:
*
dtype0
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
w
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*
_output_shapes

:

�
!Variable_1/Adam/Initializer/zerosConst*
_class
loc:@Variable_1*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_1/Adam
VariableV2*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_1*
	container *
shape
:
*
dtype0
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

:

y
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes

:

�
#Variable_1/Adam_1/Initializer/zerosConst*
_output_shapes

:
*
_class
loc:@Variable_1*
valueB
*    *
dtype0
�
Variable_1/Adam_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_1*
	container *
shape
:

�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

:

}
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*
_output_shapes

:

�
!Variable_2/Adam/Initializer/zerosConst*
_class
loc:@Variable_2*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_2/Adam
VariableV2*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_2
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_2
y
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*
_output_shapes

:

�
#Variable_2/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_2*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_2/Adam_1
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_2*
	container 
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:

}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*
_output_shapes

:

�
!Variable_3/Adam/Initializer/zerosConst*
_class
loc:@Variable_3*
valueB*    *
dtype0*
_output_shapes
:
�
Variable_3/Adam
VariableV2*
_output_shapes
:*
shared_name *
_class
loc:@Variable_3*
	container *
shape:*
dtype0
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
_output_shapes
:*
T0
�
#Variable_3/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_3*
valueB*    *
dtype0*
_output_shapes
:
�
Variable_3/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@Variable_3*
	container *
shape:
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:*
use_locking(
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *��8*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_nesterov( *
_output_shapes

:
*
use_locking( *
T0*
_class
loc:@Variable
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
T0*
_class
loc:@Variable_1*
use_nesterov( *
_output_shapes

:
*
use_locking( 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/MatMul_10_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*
use_nesterov( *
_output_shapes

:

�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_10_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*
_class
loc:@Variable_3
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@Variable
�
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam
�
initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^beta1_power/Assign^beta2_power/Assign"H�<G