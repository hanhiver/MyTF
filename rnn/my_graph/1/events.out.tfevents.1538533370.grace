       �K"	  ��	��Abrain.Event:2�D{bK     u��
	[ׄ�	��A"��
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������
*
shape:���������

p
Placeholder_1Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
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
"���P?��T?i��>�o�>�8�>�b?M�?��>f��>!*?�zw?�?�|C?w�<.��>U��>;`�=��-?r4Y?�U?m-?�{4?=q?��?�K�>�n�>�sK>�j+?�nN>�e ?&Y?��X?���>�=�=�M?��`?ߙC>9�1?32?�l�>8�?��j?8�a?!�<?^+8?���=0l)>9W�>���>�UU>>|��>��>Im�>o�?�Y?J�C=��I>��G?5ԃ>�{u?��==}�k?H>-7�>#?sxp>Y�
?��<�Ђ>>�jl?}�=>��>.�?E� ?��> ??p�%?���>1�>|:M?��?$�)>;#D>���>��N?�A?,E|?.uq?�R�>�%?��.?��>Y:"?�J�>�� :�)?xs?A�8?("@?��@?8�>��>��?���>��O>I� ?}��>t�2?*
dtype0*
_output_shapes

:

|
Variable
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

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
Variable_1/initial_valueConst*A
value8B6
"(                                        *
dtype0*
_output_shapes

:

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
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes

:

�
Variable_2/initial_valueConst*A
value8B6
"([??�֕>�n?���>o�U>��<=�`>L�<?��?w�u>*
dtype0*
_output_shapes

:

~

Variable_2
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
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
Variable_2*
_output_shapes

:
*
T0*
_class
loc:@Variable_2
e
Variable_3/initial_valueConst*
dtype0*
_output_shapes
:*
valueB*    
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
Variable_3Variable_3/initial_value*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:*
T0*
_class
loc:@Variable_3
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
concatConcatV2ReshapePlaceholder_2concat/axis*
N*
_output_shapes

:
*

Tidx0*
T0
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
TanhTanhadd*
T0*
_output_shapes

:


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
T0*
N*
_output_shapes

:
*

Tidx0
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
add_1AddMatMul_1Variable_1/read*
_output_shapes

:

*
T0
>
Tanh_1Tanhadd_1*
T0*
_output_shapes

:


`
Reshape_2/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
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
concat_2ConcatV2	Reshape_2Tanh_1concat_2/axis*

Tidx0*
T0*
N*
_output_shapes

:

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
add_2AddMatMul_2Variable_1/read*
_output_shapes

:

*
T0
>
Tanh_2Tanhadd_2*
T0*
_output_shapes

:


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
concat_3ConcatV2	Reshape_3Tanh_2concat_3/axis*

Tidx0*
T0*
N*
_output_shapes

:

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
add_3AddMatMul_3Variable_1/read*
_output_shapes

:

*
T0
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
concat_4/axisConst*
dtype0*
_output_shapes
: *
value	B :
t
concat_4ConcatV2	Reshape_4Tanh_3concat_4/axis*

Tidx0*
T0*
N*
_output_shapes

:

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
Tanh_4Tanhadd_4*
T0*
_output_shapes

:


`
Reshape_5/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_5Reshape	unstack:5Reshape_5/shape*
_output_shapes

:
*
T0*
Tshape0
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
MatMul_5MatMulconcat_5Variable/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
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
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_6Reshape	unstack:6Reshape_6/shape*
T0*
Tshape0*
_output_shapes

:

O
concat_6/axisConst*
value	B :*
dtype0*
_output_shapes
: 
t
concat_6ConcatV2	Reshape_6Tanh_5concat_6/axis*

Tidx0*
T0*
N*
_output_shapes

:

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
Tanh_6Tanhadd_6*
_output_shapes

:

*
T0
`
Reshape_7/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_7Reshape	unstack:7Reshape_7/shape*
_output_shapes

:
*
T0*
Tshape0
O
concat_7/axisConst*
_output_shapes
: *
value	B :*
dtype0
t
concat_7ConcatV2	Reshape_7Tanh_6concat_7/axis*

Tidx0*
T0*
N*
_output_shapes

:

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
add_7AddMatMul_7Variable_1/read*
_output_shapes

:

*
T0
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
concat_8/axisConst*
_output_shapes
: *
value	B :*
dtype0
t
concat_8ConcatV2	Reshape_8Tanh_7concat_8/axis*
N*
_output_shapes

:
*

Tidx0*
T0
z
MatMul_8MatMulconcat_8Variable/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
P
add_8AddMatMul_8Variable_1/read*
_output_shapes

:

*
T0
>
Tanh_8Tanhadd_8*
T0*
_output_shapes

:


`
Reshape_9/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_9Reshape	unstack:9Reshape_9/shape*
_output_shapes

:
*
T0*
Tshape0
O
concat_9/axisConst*
value	B :*
dtype0*
_output_shapes
: 
t
concat_9ConcatV2	Reshape_9Tanh_8concat_9/axis*
T0*
N*
_output_shapes

:
*

Tidx0
z
MatMul_9MatMulconcat_9Variable/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
P
add_9AddMatMul_9Variable_1/read*
T0*
_output_shapes

:


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
SubSubPlaceholder_1add_10*
_output_shapes

:
*
T0
>
SquareSquareSub*
T0*
_output_shapes

:

`
gradients/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
w
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
_output_shapes

:
*
T0
q
gradients/Square_grad/ConstConst^gradients/Fill*
_output_shapes
: *
valueB
 *   @*
dtype0
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
gradients/Sub_grad/ShapeShapePlaceholder_1*
_output_shapes
:*
T0*
out_type0
k
gradients/Sub_grad/Shape_1Const*
valueB"
      *
dtype0*
_output_shapes
:
�
(gradients/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Sub_grad/Shapegradients/Sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/Sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/Sub_grad/ReshapeReshapegradients/Sub_grad/Sumgradients/Sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/Sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/Sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
+gradients/Sub_grad/tuple/control_dependencyIdentitygradients/Sub_grad/Reshape$^gradients/Sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Sub_grad/Reshape*'
_output_shapes
:���������
�
-gradients/Sub_grad/tuple/control_dependency_1Identitygradients/Sub_grad/Reshape_1$^gradients/Sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Sub_grad/Reshape_1*
_output_shapes

:
*
T0
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
+gradients/add_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_10_grad/Shapegradients/add_10_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_10_grad/SumSum-gradients/Sub_grad/tuple/control_dependency_1+gradients/add_10_grad/BroadcastGradientArgs*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients/add_10_grad/ReshapeReshapegradients/add_10_grad/Sumgradients/add_10_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
gradients/add_10_grad/Sum_1Sum-gradients/Sub_grad/tuple/control_dependency_1-gradients/add_10_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
gradients/add_10_grad/Reshape_1Reshapegradients/add_10_grad/Sum_1gradients/add_10_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
p
&gradients/add_10_grad/tuple/group_depsNoOp^gradients/add_10_grad/Reshape ^gradients/add_10_grad/Reshape_1
�
.gradients/add_10_grad/tuple/control_dependencyIdentitygradients/add_10_grad/Reshape'^gradients/add_10_grad/tuple/group_deps*0
_class&
$"loc:@gradients/add_10_grad/Reshape*
_output_shapes

:
*
T0
�
0gradients/add_10_grad/tuple/control_dependency_1Identitygradients/add_10_grad/Reshape_1'^gradients/add_10_grad/tuple/group_deps*
_output_shapes
:*
T0*2
_class(
&$loc:@gradients/add_10_grad/Reshape_1
�
gradients/MatMul_10_grad/MatMulMatMul.gradients/add_10_grad/tuple/control_dependencyVariable_2/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b(
�
!gradients/MatMul_10_grad/MatMul_1MatMulTanh_9.gradients/add_10_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
w
)gradients/MatMul_10_grad/tuple/group_depsNoOp ^gradients/MatMul_10_grad/MatMul"^gradients/MatMul_10_grad/MatMul_1
�
1gradients/MatMul_10_grad/tuple/control_dependencyIdentitygradients/MatMul_10_grad/MatMul*^gradients/MatMul_10_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/MatMul_10_grad/MatMul*
_output_shapes

:


�
3gradients/MatMul_10_grad/tuple/control_dependency_1Identity!gradients/MatMul_10_grad/MatMul_1*^gradients/MatMul_10_grad/tuple/group_deps*4
_class*
(&loc:@gradients/MatMul_10_grad/MatMul_1*
_output_shapes

:
*
T0
�
gradients/Tanh_9_grad/TanhGradTanhGradTanh_91gradients/MatMul_10_grad/tuple/control_dependency*
T0*
_output_shapes

:


k
gradients/add_9_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
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
gradients/add_9_grad/ReshapeReshapegradients/add_9_grad/Sumgradients/add_9_grad/Shape*
_output_shapes

:

*
T0*
Tshape0
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
-gradients/add_9_grad/tuple/control_dependencyIdentitygradients/add_9_grad/Reshape&^gradients/add_9_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_9_grad/Reshape*
_output_shapes

:


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
 gradients/MatMul_9_grad/MatMul_1MatMulconcat_9-gradients/add_9_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_9_grad/tuple/group_depsNoOp^gradients/MatMul_9_grad/MatMul!^gradients/MatMul_9_grad/MatMul_1
�
0gradients/MatMul_9_grad/tuple/control_dependencyIdentitygradients/MatMul_9_grad/MatMul)^gradients/MatMul_9_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/MatMul_9_grad/MatMul
�
2gradients/MatMul_9_grad/tuple/control_dependency_1Identity gradients/MatMul_9_grad/MatMul_1)^gradients/MatMul_9_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients/MatMul_9_grad/MatMul_1
^
gradients/concat_9_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0
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
Index0*
T0*
_output_shapes

:

�
gradients/concat_9_grad/Slice_1Slice0gradients/MatMul_9_grad/tuple/control_dependency&gradients/concat_9_grad/ConcatOffset:1gradients/concat_9_grad/Shape_1*
Index0*
T0*
_output_shapes

:


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
gradients/Tanh_8_grad/TanhGradTanhGradTanh_82gradients/concat_9_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
k
gradients/add_8_grad/ShapeConst*
_output_shapes
:*
valueB"
   
   *
dtype0
m
gradients/add_8_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
�
*gradients/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_8_grad/Shapegradients/add_8_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_8_grad/SumSumgradients/Tanh_8_grad/TanhGrad*gradients/add_8_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients/add_8_grad/ReshapeReshapegradients/add_8_grad/Sumgradients/add_8_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_8_grad/Sum_1Sumgradients/Tanh_8_grad/TanhGrad,gradients/add_8_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
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
-gradients/add_8_grad/tuple/control_dependencyIdentitygradients/add_8_grad/Reshape&^gradients/add_8_grad/tuple/group_deps*
_output_shapes

:

*
T0*/
_class%
#!loc:@gradients/add_8_grad/Reshape
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
 gradients/MatMul_8_grad/MatMul_1MatMulconcat_8-gradients/add_8_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
t
(gradients/MatMul_8_grad/tuple/group_depsNoOp^gradients/MatMul_8_grad/MatMul!^gradients/MatMul_8_grad/MatMul_1
�
0gradients/MatMul_8_grad/tuple/control_dependencyIdentitygradients/MatMul_8_grad/MatMul)^gradients/MatMul_8_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_8_grad/MatMul*
_output_shapes

:

�
2gradients/MatMul_8_grad/tuple/control_dependency_1Identity gradients/MatMul_8_grad/MatMul_1)^gradients/MatMul_8_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients/MatMul_8_grad/MatMul_1
^
gradients/concat_8_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_8_grad/modFloorModconcat_8/axisgradients/concat_8_grad/Rank*
_output_shapes
: *
T0
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
_output_shapes

:
*
Index0*
T0
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
0gradients/concat_8_grad/tuple/control_dependencyIdentitygradients/concat_8_grad/Slice)^gradients/concat_8_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_8_grad/Slice*
_output_shapes

:

�
2gradients/concat_8_grad/tuple/control_dependency_1Identitygradients/concat_8_grad/Slice_1)^gradients/concat_8_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients/concat_8_grad/Slice_1
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
gradients/add_7_grad/SumSumgradients/Tanh_7_grad/TanhGrad*gradients/add_7_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
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
_output_shapes

:
*
T0*
Tshape0
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
/gradients/add_7_grad/tuple/control_dependency_1Identitygradients/add_7_grad/Reshape_1&^gradients/add_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_7_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_7_grad/MatMulMatMul-gradients/add_7_grad/tuple/control_dependencyVariable/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_7_grad/MatMul_1MatMulconcat_7-gradients/add_7_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
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
2gradients/MatMul_7_grad/tuple/control_dependency_1Identity gradients/MatMul_7_grad/MatMul_1)^gradients/MatMul_7_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_7_grad/MatMul_1*
_output_shapes

:

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
Index0*
T0*
_output_shapes

:

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
gradients/add_6_grad/SumSumgradients/Tanh_6_grad/TanhGrad*gradients/add_6_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients/add_6_grad/ReshapeReshapegradients/add_6_grad/Sumgradients/add_6_grad/Shape*
Tshape0*
_output_shapes

:

*
T0
�
gradients/add_6_grad/Sum_1Sumgradients/Tanh_6_grad/TanhGrad,gradients/add_6_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients/add_6_grad/Reshape_1Reshapegradients/add_6_grad/Sum_1gradients/add_6_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
m
%gradients/add_6_grad/tuple/group_depsNoOp^gradients/add_6_grad/Reshape^gradients/add_6_grad/Reshape_1
�
-gradients/add_6_grad/tuple/control_dependencyIdentitygradients/add_6_grad/Reshape&^gradients/add_6_grad/tuple/group_deps*
_output_shapes

:

*
T0*/
_class%
#!loc:@gradients/add_6_grad/Reshape
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
 gradients/MatMul_6_grad/MatMul_1MatMulconcat_6-gradients/add_6_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_6_grad/tuple/group_depsNoOp^gradients/MatMul_6_grad/MatMul!^gradients/MatMul_6_grad/MatMul_1
�
0gradients/MatMul_6_grad/tuple/control_dependencyIdentitygradients/MatMul_6_grad/MatMul)^gradients/MatMul_6_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/MatMul_6_grad/MatMul
�
2gradients/MatMul_6_grad/tuple/control_dependency_1Identity gradients/MatMul_6_grad/MatMul_1)^gradients/MatMul_6_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_6_grad/MatMul_1*
_output_shapes

:

^
gradients/concat_6_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_6_grad/modFloorModconcat_6/axisgradients/concat_6_grad/Rank*
_output_shapes
: *
T0
n
gradients/concat_6_grad/ShapeConst*
_output_shapes
:*
valueB"
      *
dtype0
p
gradients/concat_6_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"
   
   
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
2gradients/concat_6_grad/tuple/control_dependency_1Identitygradients/concat_6_grad/Slice_1)^gradients/concat_6_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients/concat_6_grad/Slice_1
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
*gradients/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_5_grad/Shapegradients/add_5_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_5_grad/SumSumgradients/Tanh_5_grad/TanhGrad*gradients/add_5_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients/add_5_grad/ReshapeReshapegradients/add_5_grad/Sumgradients/add_5_grad/Shape*
Tshape0*
_output_shapes

:

*
T0
�
gradients/add_5_grad/Sum_1Sumgradients/Tanh_5_grad/TanhGrad,gradients/add_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
gradients/add_5_grad/Reshape_1Reshapegradients/add_5_grad/Sum_1gradients/add_5_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

m
%gradients/add_5_grad/tuple/group_depsNoOp^gradients/add_5_grad/Reshape^gradients/add_5_grad/Reshape_1
�
-gradients/add_5_grad/tuple/control_dependencyIdentitygradients/add_5_grad/Reshape&^gradients/add_5_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_5_grad/Reshape*
_output_shapes

:


�
/gradients/add_5_grad/tuple/control_dependency_1Identitygradients/add_5_grad/Reshape_1&^gradients/add_5_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/add_5_grad/Reshape_1
�
gradients/MatMul_5_grad/MatMulMatMul-gradients/add_5_grad/tuple/control_dependencyVariable/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
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
2gradients/MatMul_5_grad/tuple/control_dependency_1Identity gradients/MatMul_5_grad/MatMul_1)^gradients/MatMul_5_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients/MatMul_5_grad/MatMul_1
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
gradients/concat_5_grad/ShapeConst*
_output_shapes
:*
valueB"
      *
dtype0
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
_output_shapes

:
*
Index0*
T0
�
gradients/concat_5_grad/Slice_1Slice0gradients/MatMul_5_grad/tuple/control_dependency&gradients/concat_5_grad/ConcatOffset:1gradients/concat_5_grad/Shape_1*
Index0*
T0*
_output_shapes

:


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
gradients/Tanh_4_grad/TanhGradTanhGradTanh_42gradients/concat_5_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
k
gradients/add_4_grad/ShapeConst*
_output_shapes
:*
valueB"
   
   *
dtype0
m
gradients/add_4_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
�
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*
_output_shapes

:

*
T0*/
_class%
#!loc:@gradients/add_4_grad/Reshape
�
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*
_output_shapes

:
*
T0
�
gradients/MatMul_4_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_4_grad/MatMul_1MatMulconcat_4-gradients/add_4_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
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
gradients/concat_4_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
u
gradients/concat_4_grad/modFloorModconcat_4/axisgradients/concat_4_grad/Rank*
_output_shapes
: *
T0
n
gradients/concat_4_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
p
gradients/concat_4_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_4_grad/ConcatOffsetConcatOffsetgradients/concat_4_grad/modgradients/concat_4_grad/Shapegradients/concat_4_grad/Shape_1*
N* 
_output_shapes
::
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
2gradients/concat_4_grad/tuple/control_dependency_1Identitygradients/concat_4_grad/Slice_1)^gradients/concat_4_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_4_grad/Slice_1*
_output_shapes

:


�
gradients/Tanh_3_grad/TanhGradTanhGradTanh_32gradients/concat_4_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


k
gradients/add_3_grad/ShapeConst*
_output_shapes
:*
valueB"
   
   *
dtype0
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
gradients/add_3_grad/SumSumgradients/Tanh_3_grad/TanhGrad*gradients/add_3_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
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
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
_output_shapes

:

*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape
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
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*
_output_shapes

:
*
T0
�
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1
^
gradients/concat_3_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
u
gradients/concat_3_grad/modFloorModconcat_3/axisgradients/concat_3_grad/Rank*
T0*
_output_shapes
: 
n
gradients/concat_3_grad/ShapeConst*
_output_shapes
:*
valueB"
      *
dtype0
p
gradients/concat_3_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"
   
   
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
0gradients/concat_3_grad/tuple/control_dependencyIdentitygradients/concat_3_grad/Slice)^gradients/concat_3_grad/tuple/group_deps*0
_class&
$"loc:@gradients/concat_3_grad/Slice*
_output_shapes

:
*
T0
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
gradients/Tanh_2_grad/TanhGradTanhGradTanh_22gradients/concat_3_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
k
gradients/add_2_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
m
gradients/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB"   
   *
dtype0
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_2_grad/SumSumgradients/Tanh_2_grad/TanhGrad*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


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
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
_output_shapes

:

*
T0
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1
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
gradients/concat_2_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_2_grad/modFloorModconcat_2/axisgradients/concat_2_grad/Rank*
_output_shapes
: *
T0
n
gradients/concat_2_grad/ShapeConst*
_output_shapes
:*
valueB"
      *
dtype0
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
gradients/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB"   
   *
dtype0
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Tanh_1_grad/TanhGrad*gradients/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
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
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
_output_shapes

:

*
T0
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes

:
*
T0
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMulconcat_1-gradients/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
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
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:

^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_1_grad/modFloorModconcat_1/axisgradients/concat_1_grad/Rank*
_output_shapes
: *
T0
n
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
valueB"
      *
dtype0
p
gradients/concat_1_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/modgradients/concat_1_grad/Shapegradients/concat_1_grad/Shape_1*
N* 
_output_shapes
::
�
gradients/concat_1_grad/SliceSlice0gradients/MatMul_1_grad/tuple/control_dependency$gradients/concat_1_grad/ConcatOffsetgradients/concat_1_grad/Shape*
Index0*
T0*
_output_shapes

:

�
gradients/concat_1_grad/Slice_1Slice0gradients/MatMul_1_grad/tuple/control_dependency&gradients/concat_1_grad/ConcatOffset:1gradients/concat_1_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
r
(gradients/concat_1_grad/tuple/group_depsNoOp^gradients/concat_1_grad/Slice ^gradients/concat_1_grad/Slice_1
�
0gradients/concat_1_grad/tuple/control_dependencyIdentitygradients/concat_1_grad/Slice)^gradients/concat_1_grad/tuple/group_deps*
_output_shapes

:
*
T0*0
_class&
$"loc:@gradients/concat_1_grad/Slice
�
2gradients/concat_1_grad/tuple/control_dependency_1Identitygradients/concat_1_grad/Slice_1)^gradients/concat_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_1_grad/Slice_1*
_output_shapes

:


�
gradients/Tanh_grad/TanhGradTanhGradTanh2gradients/concat_1_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
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
gradients/add_grad/SumSumgradients/Tanh_grad/TanhGrad(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
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
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

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
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
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
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
_output_shapes

:
*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:
*
T0
�
gradients/AddNAddN/gradients/add_9_grad/tuple/control_dependency_1/gradients/add_8_grad/tuple/control_dependency_1/gradients/add_7_grad/tuple/control_dependency_1/gradients/add_6_grad/tuple/control_dependency_1/gradients/add_5_grad/tuple/control_dependency_1/gradients/add_4_grad/tuple/control_dependency_1/gradients/add_3_grad/tuple/control_dependency_1/gradients/add_2_grad/tuple/control_dependency_1/gradients/add_1_grad/tuple/control_dependency_1-gradients/add_grad/tuple/control_dependency_1*1
_class'
%#loc:@gradients/add_9_grad/Reshape_1*
N
*
_output_shapes

:
*
T0
�
gradients/AddN_1AddN2gradients/MatMul_9_grad/tuple/control_dependency_12gradients/MatMul_8_grad/tuple/control_dependency_12gradients/MatMul_7_grad/tuple/control_dependency_12gradients/MatMul_6_grad/tuple/control_dependency_12gradients/MatMul_5_grad/tuple/control_dependency_12gradients/MatMul_4_grad/tuple/control_dependency_12gradients/MatMul_3_grad/tuple/control_dependency_12gradients/MatMul_2_grad/tuple/control_dependency_12gradients/MatMul_1_grad/tuple/control_dependency_10gradients/MatMul_grad/tuple/control_dependency_1*
N
*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients/MatMul_9_grad/MatMul_1
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
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable*
	container *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
loc:@Variable
{
beta2_power/initial_valueConst*
_class
loc:@Variable*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable*
	container *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
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
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable
s
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*
_output_shapes

:

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
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable*
	container *
shape
:

�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
w
Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
_output_shapes

:
*
T0
�
!Variable_1/Adam/Initializer/zerosConst*
_output_shapes

:
*
_class
loc:@Variable_1*
valueB
*    *
dtype0
�
Variable_1/Adam
VariableV2*
shared_name *
_class
loc:@Variable_1*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(
y
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes

:
*
T0*
_class
loc:@Variable_1
�
#Variable_1/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_1*
valueB
*    *
dtype0*
_output_shapes

:

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
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
_output_shapes

:
*
T0
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
VariableV2*
_class
loc:@Variable_2*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
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
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_2*
	container *
shape
:

�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:
*
use_locking(
}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_output_shapes

:
*
T0*
_class
loc:@Variable_2
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
VariableV2*
shared_name *
_class
loc:@Variable_3*
	container *
shape:*
dtype0*
_output_shapes
:
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:
u
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes
:
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
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
_output_shapes
:*
T0
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *��8*
dtype0
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
Adam/beta2Adam/epsilongradients/AddN_1*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( *
_output_shapes

:

�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( *
_output_shapes

:

�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/MatMul_10_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_2*
use_nesterov( *
_output_shapes

:
*
use_locking( 
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_10_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_3*
use_nesterov( *
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
�
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam
�
initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^beta1_power/Assign^beta2_power/Assign
p
Placeholder_3Placeholder*
shape:���������
*
dtype0*'
_output_shapes
:���������

p
Placeholder_4Placeholder*'
_output_shapes
:���������*
shape:���������*
dtype0
^
Placeholder_5Placeholder*
_output_shapes

:

*
shape
:

*
dtype0
�
Variable_4/initial_valueConst*�
value�B�
"�`p�>>?Ջ�>���>ĕ">~�"?Xa?���>��=i��>�>�*>��>>X/\?�I?4?�a�>)N�=�lb?5�>?�_>�?��F?nd�>3�?4t�>�ɻ>�G=l?*�?ب�={w?��=e�>�Lk<�/�=�w?.�>�>�9?��1=�>��U?���>C��>1��=�%�>bj?�Z?�x'>c�->bA?y�:?|P.?pq�>%@>.`?��c?H& >
�>��>�B?�W?�k8?->���>��?�c?��}?�>��h?>e?��t?�?!V?*�>��>�v?��S>fcw?O|>�J?=��>��>�?0=�
?�K=[O+?��>~?vv�>ꨁ>Ԡ�>]��>~�P?>�1?2�>��W>�O?�w�>'��>
�(?���>��F?n�>pz?41�>�M?�|?*
dtype0*
_output_shapes

:

~

Variable_4
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
Variable_4/AssignAssign
Variable_4Variable_4/initial_value*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
o
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:

�
Variable_5/initial_valueConst*
dtype0*
_output_shapes

:
*A
value8B6
"(                                        
~

Variable_5
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
�
Variable_5/AssignAssign
Variable_5Variable_5/initial_value*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes

:

o
Variable_5/readIdentity
Variable_5*
_output_shapes

:
*
T0*
_class
loc:@Variable_5
�
Variable_6/initial_valueConst*
_output_shapes

:
*A
value8B6
"(?	+?�+?��?�1?��i?t�A>��n?�4&?8�?67?*
dtype0
~

Variable_6
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
Variable_6/AssignAssign
Variable_6Variable_6/initial_value*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
o
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes

:

e
Variable_7/initial_valueConst*
valueB*    *
dtype0*
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
Variable_7/AssignAssign
Variable_7Variable_7/initial_value*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
_output_shapes
:*
T0
�
	unstack_1UnpackPlaceholder_3*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*	
num
*
T0*

axis
a
Reshape_10/shapeConst*
_output_shapes
:*
valueB"
      *
dtype0
i

Reshape_10Reshape	unstack_1Reshape_10/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_10/axisConst*
value	B :*
dtype0*
_output_shapes
: 
~
	concat_10ConcatV2
Reshape_10Placeholder_5concat_10/axis*
T0*
N*
_output_shapes

:
*

Tidx0
~
	MatMul_11MatMul	concat_10Variable_4/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
R
add_11Add	MatMul_11Variable_5/read*
T0*
_output_shapes

:


@
Tanh_10Tanhadd_11*
_output_shapes

:

*
T0
a
Reshape_11/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_11Reshapeunstack_1:1Reshape_11/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_11/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_11ConcatV2
Reshape_11Tanh_10concat_11/axis*
T0*
N*
_output_shapes

:
*

Tidx0
~
	MatMul_12MatMul	concat_11Variable_4/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
R
add_12Add	MatMul_12Variable_5/read*
T0*
_output_shapes

:


@
Tanh_11Tanhadd_12*
T0*
_output_shapes

:


a
Reshape_12/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_12Reshapeunstack_1:2Reshape_12/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_12/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_12ConcatV2
Reshape_12Tanh_11concat_12/axis*

Tidx0*
T0*
N*
_output_shapes

:

~
	MatMul_13MatMul	concat_12Variable_4/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
R
add_13Add	MatMul_13Variable_5/read*
_output_shapes

:

*
T0
@
Tanh_12Tanhadd_13*
T0*
_output_shapes

:


a
Reshape_13/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_13Reshapeunstack_1:3Reshape_13/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_13/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_13ConcatV2
Reshape_13Tanh_12concat_13/axis*

Tidx0*
T0*
N*
_output_shapes

:

~
	MatMul_14MatMul	concat_13Variable_4/read*
transpose_b( *
T0*
_output_shapes

:

*
transpose_a( 
R
add_14Add	MatMul_14Variable_5/read*
T0*
_output_shapes

:


@
Tanh_13Tanhadd_14*
_output_shapes

:

*
T0
a
Reshape_14/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_14Reshapeunstack_1:4Reshape_14/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_14/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_14ConcatV2
Reshape_14Tanh_13concat_14/axis*
N*
_output_shapes

:
*

Tidx0*
T0
~
	MatMul_15MatMul	concat_14Variable_4/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
R
add_15Add	MatMul_15Variable_5/read*
T0*
_output_shapes

:


@
Tanh_14Tanhadd_15*
_output_shapes

:

*
T0
a
Reshape_15/shapeConst*
_output_shapes
:*
valueB"
      *
dtype0
k

Reshape_15Reshapeunstack_1:5Reshape_15/shape*
Tshape0*
_output_shapes

:
*
T0
P
concat_15/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_15ConcatV2
Reshape_15Tanh_14concat_15/axis*

Tidx0*
T0*
N*
_output_shapes

:

~
	MatMul_16MatMul	concat_15Variable_4/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
R
add_16Add	MatMul_16Variable_5/read*
_output_shapes

:

*
T0
@
Tanh_15Tanhadd_16*
T0*
_output_shapes

:


a
Reshape_16/shapeConst*
_output_shapes
:*
valueB"
      *
dtype0
k

Reshape_16Reshapeunstack_1:6Reshape_16/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_16/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_16ConcatV2
Reshape_16Tanh_15concat_16/axis*
N*
_output_shapes

:
*

Tidx0*
T0
~
	MatMul_17MatMul	concat_16Variable_4/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
R
add_17Add	MatMul_17Variable_5/read*
_output_shapes

:

*
T0
@
Tanh_16Tanhadd_17*
T0*
_output_shapes

:


a
Reshape_17/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_17Reshapeunstack_1:7Reshape_17/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_17/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_17ConcatV2
Reshape_17Tanh_16concat_17/axis*
T0*
N*
_output_shapes

:
*

Tidx0
~
	MatMul_18MatMul	concat_17Variable_4/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
R
add_18Add	MatMul_18Variable_5/read*
T0*
_output_shapes

:


@
Tanh_17Tanhadd_18*
T0*
_output_shapes

:


a
Reshape_18/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_18Reshapeunstack_1:8Reshape_18/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_18/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_18ConcatV2
Reshape_18Tanh_17concat_18/axis*
N*
_output_shapes

:
*

Tidx0*
T0
~
	MatMul_19MatMul	concat_18Variable_4/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
R
add_19Add	MatMul_19Variable_5/read*
_output_shapes

:

*
T0
@
Tanh_18Tanhadd_19*
_output_shapes

:

*
T0
a
Reshape_19/shapeConst*
_output_shapes
:*
valueB"
      *
dtype0
k

Reshape_19Reshapeunstack_1:9Reshape_19/shape*
_output_shapes

:
*
T0*
Tshape0
P
concat_19/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_19ConcatV2
Reshape_19Tanh_18concat_19/axis*

Tidx0*
T0*
N*
_output_shapes

:

~
	MatMul_20MatMul	concat_19Variable_4/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
R
add_20Add	MatMul_20Variable_5/read*
T0*
_output_shapes

:


@
Tanh_19Tanhadd_20*
T0*
_output_shapes

:


|
	MatMul_21MatMulTanh_19Variable_6/read*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a( 
R
add_21Add	MatMul_21Variable_7/read*
_output_shapes

:
*
T0
L
Sub_1SubPlaceholder_4add_21*
T0*
_output_shapes

:

B
Square_1SquareSub_1*
_output_shapes

:
*
T0
b
gradients_1/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
Z
gradients_1/grad_ys_0Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
}
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*

index_type0*
_output_shapes

:
*
T0
w
gradients_1/Square_1_grad/ConstConst^gradients_1/Fill*
valueB
 *   @*
dtype0*
_output_shapes
: 
u
gradients_1/Square_1_grad/MulMulSub_1gradients_1/Square_1_grad/Const*
T0*
_output_shapes

:

�
gradients_1/Square_1_grad/Mul_1Mulgradients_1/Fillgradients_1/Square_1_grad/Mul*
_output_shapes

:
*
T0
i
gradients_1/Sub_1_grad/ShapeShapePlaceholder_4*
T0*
out_type0*
_output_shapes
:
o
gradients_1/Sub_1_grad/Shape_1Const*
valueB"
      *
dtype0*
_output_shapes
:
�
,gradients_1/Sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Sub_1_grad/Shapegradients_1/Sub_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/Sub_1_grad/SumSumgradients_1/Square_1_grad/Mul_1,gradients_1/Sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients_1/Sub_1_grad/ReshapeReshapegradients_1/Sub_1_grad/Sumgradients_1/Sub_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients_1/Sub_1_grad/Sum_1Sumgradients_1/Square_1_grad/Mul_1.gradients_1/Sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
b
gradients_1/Sub_1_grad/NegNeggradients_1/Sub_1_grad/Sum_1*
_output_shapes
:*
T0
�
 gradients_1/Sub_1_grad/Reshape_1Reshapegradients_1/Sub_1_grad/Neggradients_1/Sub_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

s
'gradients_1/Sub_1_grad/tuple/group_depsNoOp^gradients_1/Sub_1_grad/Reshape!^gradients_1/Sub_1_grad/Reshape_1
�
/gradients_1/Sub_1_grad/tuple/control_dependencyIdentitygradients_1/Sub_1_grad/Reshape(^gradients_1/Sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/Sub_1_grad/Reshape*'
_output_shapes
:���������
�
1gradients_1/Sub_1_grad/tuple/control_dependency_1Identity gradients_1/Sub_1_grad/Reshape_1(^gradients_1/Sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/Sub_1_grad/Reshape_1*
_output_shapes

:

n
gradients_1/add_21_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
i
gradients_1/add_21_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
-gradients_1/add_21_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_21_grad/Shapegradients_1/add_21_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_21_grad/SumSum1gradients_1/Sub_1_grad/tuple/control_dependency_1-gradients_1/add_21_grad/BroadcastGradientArgs*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients_1/add_21_grad/ReshapeReshapegradients_1/add_21_grad/Sumgradients_1/add_21_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
gradients_1/add_21_grad/Sum_1Sum1gradients_1/Sub_1_grad/tuple/control_dependency_1/gradients_1/add_21_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
!gradients_1/add_21_grad/Reshape_1Reshapegradients_1/add_21_grad/Sum_1gradients_1/add_21_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
v
(gradients_1/add_21_grad/tuple/group_depsNoOp ^gradients_1/add_21_grad/Reshape"^gradients_1/add_21_grad/Reshape_1
�
0gradients_1/add_21_grad/tuple/control_dependencyIdentitygradients_1/add_21_grad/Reshape)^gradients_1/add_21_grad/tuple/group_deps*
_output_shapes

:
*
T0*2
_class(
&$loc:@gradients_1/add_21_grad/Reshape
�
2gradients_1/add_21_grad/tuple/control_dependency_1Identity!gradients_1/add_21_grad/Reshape_1)^gradients_1/add_21_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_21_grad/Reshape_1*
_output_shapes
:
�
!gradients_1/MatMul_21_grad/MatMulMatMul0gradients_1/add_21_grad/tuple/control_dependencyVariable_6/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_21_grad/MatMul_1MatMulTanh_190gradients_1/add_21_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
}
+gradients_1/MatMul_21_grad/tuple/group_depsNoOp"^gradients_1/MatMul_21_grad/MatMul$^gradients_1/MatMul_21_grad/MatMul_1
�
3gradients_1/MatMul_21_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_21_grad/MatMul,^gradients_1/MatMul_21_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_21_grad/MatMul*
_output_shapes

:


�
5gradients_1/MatMul_21_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_21_grad/MatMul_1,^gradients_1/MatMul_21_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_21_grad/MatMul_1*
_output_shapes

:

�
!gradients_1/Tanh_19_grad/TanhGradTanhGradTanh_193gradients_1/MatMul_21_grad/tuple/control_dependency*
_output_shapes

:

*
T0
n
gradients_1/add_20_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
   
   
p
gradients_1/add_20_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_20_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_20_grad/Shapegradients_1/add_20_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_20_grad/SumSum!gradients_1/Tanh_19_grad/TanhGrad-gradients_1/add_20_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients_1/add_20_grad/ReshapeReshapegradients_1/add_20_grad/Sumgradients_1/add_20_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_20_grad/Sum_1Sum!gradients_1/Tanh_19_grad/TanhGrad/gradients_1/add_20_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_20_grad/Reshape_1Reshapegradients_1/add_20_grad/Sum_1gradients_1/add_20_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_20_grad/tuple/group_depsNoOp ^gradients_1/add_20_grad/Reshape"^gradients_1/add_20_grad/Reshape_1
�
0gradients_1/add_20_grad/tuple/control_dependencyIdentitygradients_1/add_20_grad/Reshape)^gradients_1/add_20_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_20_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_20_grad/tuple/control_dependency_1Identity!gradients_1/add_20_grad/Reshape_1)^gradients_1/add_20_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_20_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_20_grad/MatMulMatMul0gradients_1/add_20_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_20_grad/MatMul_1MatMul	concat_190gradients_1/add_20_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
}
+gradients_1/MatMul_20_grad/tuple/group_depsNoOp"^gradients_1/MatMul_20_grad/MatMul$^gradients_1/MatMul_20_grad/MatMul_1
�
3gradients_1/MatMul_20_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_20_grad/MatMul,^gradients_1/MatMul_20_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_20_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_20_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_20_grad/MatMul_1,^gradients_1/MatMul_20_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_20_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_19_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_19_grad/modFloorModconcat_19/axisgradients_1/concat_19_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_19_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
s
"gradients_1/concat_19_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_19_grad/ConcatOffsetConcatOffsetgradients_1/concat_19_grad/mod gradients_1/concat_19_grad/Shape"gradients_1/concat_19_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_19_grad/SliceSlice3gradients_1/MatMul_20_grad/tuple/control_dependency'gradients_1/concat_19_grad/ConcatOffset gradients_1/concat_19_grad/Shape*
_output_shapes

:
*
Index0*
T0
�
"gradients_1/concat_19_grad/Slice_1Slice3gradients_1/MatMul_20_grad/tuple/control_dependency)gradients_1/concat_19_grad/ConcatOffset:1"gradients_1/concat_19_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
{
+gradients_1/concat_19_grad/tuple/group_depsNoOp!^gradients_1/concat_19_grad/Slice#^gradients_1/concat_19_grad/Slice_1
�
3gradients_1/concat_19_grad/tuple/control_dependencyIdentity gradients_1/concat_19_grad/Slice,^gradients_1/concat_19_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/concat_19_grad/Slice*
_output_shapes

:

�
5gradients_1/concat_19_grad/tuple/control_dependency_1Identity"gradients_1/concat_19_grad/Slice_1,^gradients_1/concat_19_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/concat_19_grad/Slice_1*
_output_shapes

:


�
!gradients_1/Tanh_18_grad/TanhGradTanhGradTanh_185gradients_1/concat_19_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
n
gradients_1/add_19_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
   
   
p
gradients_1/add_19_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_19_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_19_grad/Shapegradients_1/add_19_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_19_grad/SumSum!gradients_1/Tanh_18_grad/TanhGrad-gradients_1/add_19_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients_1/add_19_grad/ReshapeReshapegradients_1/add_19_grad/Sumgradients_1/add_19_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_19_grad/Sum_1Sum!gradients_1/Tanh_18_grad/TanhGrad/gradients_1/add_19_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_19_grad/Reshape_1Reshapegradients_1/add_19_grad/Sum_1gradients_1/add_19_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_19_grad/tuple/group_depsNoOp ^gradients_1/add_19_grad/Reshape"^gradients_1/add_19_grad/Reshape_1
�
0gradients_1/add_19_grad/tuple/control_dependencyIdentitygradients_1/add_19_grad/Reshape)^gradients_1/add_19_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_19_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_19_grad/tuple/control_dependency_1Identity!gradients_1/add_19_grad/Reshape_1)^gradients_1/add_19_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_19_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_19_grad/MatMulMatMul0gradients_1/add_19_grad/tuple/control_dependencyVariable_4/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
�
#gradients_1/MatMul_19_grad/MatMul_1MatMul	concat_180gradients_1/add_19_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_19_grad/tuple/group_depsNoOp"^gradients_1/MatMul_19_grad/MatMul$^gradients_1/MatMul_19_grad/MatMul_1
�
3gradients_1/MatMul_19_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_19_grad/MatMul,^gradients_1/MatMul_19_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_19_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_19_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_19_grad/MatMul_1,^gradients_1/MatMul_19_grad/tuple/group_deps*6
_class,
*(loc:@gradients_1/MatMul_19_grad/MatMul_1*
_output_shapes

:
*
T0
a
gradients_1/concat_18_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_18_grad/modFloorModconcat_18/axisgradients_1/concat_18_grad/Rank*
_output_shapes
: *
T0
q
 gradients_1/concat_18_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_18_grad/Shape_1Const*
_output_shapes
:*
valueB"
   
   *
dtype0
�
'gradients_1/concat_18_grad/ConcatOffsetConcatOffsetgradients_1/concat_18_grad/mod gradients_1/concat_18_grad/Shape"gradients_1/concat_18_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_18_grad/SliceSlice3gradients_1/MatMul_19_grad/tuple/control_dependency'gradients_1/concat_18_grad/ConcatOffset gradients_1/concat_18_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_18_grad/Slice_1Slice3gradients_1/MatMul_19_grad/tuple/control_dependency)gradients_1/concat_18_grad/ConcatOffset:1"gradients_1/concat_18_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_18_grad/tuple/group_depsNoOp!^gradients_1/concat_18_grad/Slice#^gradients_1/concat_18_grad/Slice_1
�
3gradients_1/concat_18_grad/tuple/control_dependencyIdentity gradients_1/concat_18_grad/Slice,^gradients_1/concat_18_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/concat_18_grad/Slice*
_output_shapes

:

�
5gradients_1/concat_18_grad/tuple/control_dependency_1Identity"gradients_1/concat_18_grad/Slice_1,^gradients_1/concat_18_grad/tuple/group_deps*
_output_shapes

:

*
T0*5
_class+
)'loc:@gradients_1/concat_18_grad/Slice_1
�
!gradients_1/Tanh_17_grad/TanhGradTanhGradTanh_175gradients_1/concat_18_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_18_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_18_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_18_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_18_grad/Shapegradients_1/add_18_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_18_grad/SumSum!gradients_1/Tanh_17_grad/TanhGrad-gradients_1/add_18_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients_1/add_18_grad/ReshapeReshapegradients_1/add_18_grad/Sumgradients_1/add_18_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_18_grad/Sum_1Sum!gradients_1/Tanh_17_grad/TanhGrad/gradients_1/add_18_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
!gradients_1/add_18_grad/Reshape_1Reshapegradients_1/add_18_grad/Sum_1gradients_1/add_18_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_18_grad/tuple/group_depsNoOp ^gradients_1/add_18_grad/Reshape"^gradients_1/add_18_grad/Reshape_1
�
0gradients_1/add_18_grad/tuple/control_dependencyIdentitygradients_1/add_18_grad/Reshape)^gradients_1/add_18_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_18_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_18_grad/tuple/control_dependency_1Identity!gradients_1/add_18_grad/Reshape_1)^gradients_1/add_18_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_18_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_18_grad/MatMulMatMul0gradients_1/add_18_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_18_grad/MatMul_1MatMul	concat_170gradients_1/add_18_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
}
+gradients_1/MatMul_18_grad/tuple/group_depsNoOp"^gradients_1/MatMul_18_grad/MatMul$^gradients_1/MatMul_18_grad/MatMul_1
�
3gradients_1/MatMul_18_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_18_grad/MatMul,^gradients_1/MatMul_18_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_18_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_18_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_18_grad/MatMul_1,^gradients_1/MatMul_18_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_18_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_17_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_17_grad/modFloorModconcat_17/axisgradients_1/concat_17_grad/Rank*
_output_shapes
: *
T0
q
 gradients_1/concat_17_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_17_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_17_grad/ConcatOffsetConcatOffsetgradients_1/concat_17_grad/mod gradients_1/concat_17_grad/Shape"gradients_1/concat_17_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_17_grad/SliceSlice3gradients_1/MatMul_18_grad/tuple/control_dependency'gradients_1/concat_17_grad/ConcatOffset gradients_1/concat_17_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_17_grad/Slice_1Slice3gradients_1/MatMul_18_grad/tuple/control_dependency)gradients_1/concat_17_grad/ConcatOffset:1"gradients_1/concat_17_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_17_grad/tuple/group_depsNoOp!^gradients_1/concat_17_grad/Slice#^gradients_1/concat_17_grad/Slice_1
�
3gradients_1/concat_17_grad/tuple/control_dependencyIdentity gradients_1/concat_17_grad/Slice,^gradients_1/concat_17_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients_1/concat_17_grad/Slice
�
5gradients_1/concat_17_grad/tuple/control_dependency_1Identity"gradients_1/concat_17_grad/Slice_1,^gradients_1/concat_17_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/concat_17_grad/Slice_1*
_output_shapes

:


�
!gradients_1/Tanh_16_grad/TanhGradTanhGradTanh_165gradients_1/concat_17_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
n
gradients_1/add_17_grad/ShapeConst*
_output_shapes
:*
valueB"
   
   *
dtype0
p
gradients_1/add_17_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_17_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_17_grad/Shapegradients_1/add_17_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_17_grad/SumSum!gradients_1/Tanh_16_grad/TanhGrad-gradients_1/add_17_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients_1/add_17_grad/ReshapeReshapegradients_1/add_17_grad/Sumgradients_1/add_17_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_17_grad/Sum_1Sum!gradients_1/Tanh_16_grad/TanhGrad/gradients_1/add_17_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
!gradients_1/add_17_grad/Reshape_1Reshapegradients_1/add_17_grad/Sum_1gradients_1/add_17_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
v
(gradients_1/add_17_grad/tuple/group_depsNoOp ^gradients_1/add_17_grad/Reshape"^gradients_1/add_17_grad/Reshape_1
�
0gradients_1/add_17_grad/tuple/control_dependencyIdentitygradients_1/add_17_grad/Reshape)^gradients_1/add_17_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_17_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_17_grad/tuple/control_dependency_1Identity!gradients_1/add_17_grad/Reshape_1)^gradients_1/add_17_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_17_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_17_grad/MatMulMatMul0gradients_1/add_17_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_17_grad/MatMul_1MatMul	concat_160gradients_1/add_17_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
}
+gradients_1/MatMul_17_grad/tuple/group_depsNoOp"^gradients_1/MatMul_17_grad/MatMul$^gradients_1/MatMul_17_grad/MatMul_1
�
3gradients_1/MatMul_17_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_17_grad/MatMul,^gradients_1/MatMul_17_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_17_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_17_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_17_grad/MatMul_1,^gradients_1/MatMul_17_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_17_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_16_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
|
gradients_1/concat_16_grad/modFloorModconcat_16/axisgradients_1/concat_16_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_16_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_16_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_16_grad/ConcatOffsetConcatOffsetgradients_1/concat_16_grad/mod gradients_1/concat_16_grad/Shape"gradients_1/concat_16_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_16_grad/SliceSlice3gradients_1/MatMul_17_grad/tuple/control_dependency'gradients_1/concat_16_grad/ConcatOffset gradients_1/concat_16_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_16_grad/Slice_1Slice3gradients_1/MatMul_17_grad/tuple/control_dependency)gradients_1/concat_16_grad/ConcatOffset:1"gradients_1/concat_16_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_16_grad/tuple/group_depsNoOp!^gradients_1/concat_16_grad/Slice#^gradients_1/concat_16_grad/Slice_1
�
3gradients_1/concat_16_grad/tuple/control_dependencyIdentity gradients_1/concat_16_grad/Slice,^gradients_1/concat_16_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients_1/concat_16_grad/Slice
�
5gradients_1/concat_16_grad/tuple/control_dependency_1Identity"gradients_1/concat_16_grad/Slice_1,^gradients_1/concat_16_grad/tuple/group_deps*
_output_shapes

:

*
T0*5
_class+
)'loc:@gradients_1/concat_16_grad/Slice_1
�
!gradients_1/Tanh_15_grad/TanhGradTanhGradTanh_155gradients_1/concat_16_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_16_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_16_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
�
-gradients_1/add_16_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_16_grad/Shapegradients_1/add_16_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_16_grad/SumSum!gradients_1/Tanh_15_grad/TanhGrad-gradients_1/add_16_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients_1/add_16_grad/ReshapeReshapegradients_1/add_16_grad/Sumgradients_1/add_16_grad/Shape*
Tshape0*
_output_shapes

:

*
T0
�
gradients_1/add_16_grad/Sum_1Sum!gradients_1/Tanh_15_grad/TanhGrad/gradients_1/add_16_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_16_grad/Reshape_1Reshapegradients_1/add_16_grad/Sum_1gradients_1/add_16_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0
v
(gradients_1/add_16_grad/tuple/group_depsNoOp ^gradients_1/add_16_grad/Reshape"^gradients_1/add_16_grad/Reshape_1
�
0gradients_1/add_16_grad/tuple/control_dependencyIdentitygradients_1/add_16_grad/Reshape)^gradients_1/add_16_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_16_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_16_grad/tuple/control_dependency_1Identity!gradients_1/add_16_grad/Reshape_1)^gradients_1/add_16_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/add_16_grad/Reshape_1*
_output_shapes

:
*
T0
�
!gradients_1/MatMul_16_grad/MatMulMatMul0gradients_1/add_16_grad/tuple/control_dependencyVariable_4/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
�
#gradients_1/MatMul_16_grad/MatMul_1MatMul	concat_150gradients_1/add_16_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_16_grad/tuple/group_depsNoOp"^gradients_1/MatMul_16_grad/MatMul$^gradients_1/MatMul_16_grad/MatMul_1
�
3gradients_1/MatMul_16_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_16_grad/MatMul,^gradients_1/MatMul_16_grad/tuple/group_deps*
_output_shapes

:
*
T0*4
_class*
(&loc:@gradients_1/MatMul_16_grad/MatMul
�
5gradients_1/MatMul_16_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_16_grad/MatMul_1,^gradients_1/MatMul_16_grad/tuple/group_deps*
_output_shapes

:
*
T0*6
_class,
*(loc:@gradients_1/MatMul_16_grad/MatMul_1
a
gradients_1/concat_15_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_15_grad/modFloorModconcat_15/axisgradients_1/concat_15_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_15_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_15_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_15_grad/ConcatOffsetConcatOffsetgradients_1/concat_15_grad/mod gradients_1/concat_15_grad/Shape"gradients_1/concat_15_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_15_grad/SliceSlice3gradients_1/MatMul_16_grad/tuple/control_dependency'gradients_1/concat_15_grad/ConcatOffset gradients_1/concat_15_grad/Shape*
_output_shapes

:
*
Index0*
T0
�
"gradients_1/concat_15_grad/Slice_1Slice3gradients_1/MatMul_16_grad/tuple/control_dependency)gradients_1/concat_15_grad/ConcatOffset:1"gradients_1/concat_15_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
{
+gradients_1/concat_15_grad/tuple/group_depsNoOp!^gradients_1/concat_15_grad/Slice#^gradients_1/concat_15_grad/Slice_1
�
3gradients_1/concat_15_grad/tuple/control_dependencyIdentity gradients_1/concat_15_grad/Slice,^gradients_1/concat_15_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/concat_15_grad/Slice*
_output_shapes

:
*
T0
�
5gradients_1/concat_15_grad/tuple/control_dependency_1Identity"gradients_1/concat_15_grad/Slice_1,^gradients_1/concat_15_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/concat_15_grad/Slice_1*
_output_shapes

:


�
!gradients_1/Tanh_14_grad/TanhGradTanhGradTanh_145gradients_1/concat_15_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
n
gradients_1/add_15_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
   
   
p
gradients_1/add_15_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_15_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_15_grad/Shapegradients_1/add_15_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_15_grad/SumSum!gradients_1/Tanh_14_grad/TanhGrad-gradients_1/add_15_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients_1/add_15_grad/ReshapeReshapegradients_1/add_15_grad/Sumgradients_1/add_15_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_15_grad/Sum_1Sum!gradients_1/Tanh_14_grad/TanhGrad/gradients_1/add_15_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_15_grad/Reshape_1Reshapegradients_1/add_15_grad/Sum_1gradients_1/add_15_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_15_grad/tuple/group_depsNoOp ^gradients_1/add_15_grad/Reshape"^gradients_1/add_15_grad/Reshape_1
�
0gradients_1/add_15_grad/tuple/control_dependencyIdentitygradients_1/add_15_grad/Reshape)^gradients_1/add_15_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_15_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_15_grad/tuple/control_dependency_1Identity!gradients_1/add_15_grad/Reshape_1)^gradients_1/add_15_grad/tuple/group_deps*
_output_shapes

:
*
T0*4
_class*
(&loc:@gradients_1/add_15_grad/Reshape_1
�
!gradients_1/MatMul_15_grad/MatMulMatMul0gradients_1/add_15_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
T0*
_output_shapes

:
*
transpose_a( 
�
#gradients_1/MatMul_15_grad/MatMul_1MatMul	concat_140gradients_1/add_15_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_15_grad/tuple/group_depsNoOp"^gradients_1/MatMul_15_grad/MatMul$^gradients_1/MatMul_15_grad/MatMul_1
�
3gradients_1/MatMul_15_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_15_grad/MatMul,^gradients_1/MatMul_15_grad/tuple/group_deps*
_output_shapes

:
*
T0*4
_class*
(&loc:@gradients_1/MatMul_15_grad/MatMul
�
5gradients_1/MatMul_15_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_15_grad/MatMul_1,^gradients_1/MatMul_15_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_15_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_14_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_14_grad/modFloorModconcat_14/axisgradients_1/concat_14_grad/Rank*
_output_shapes
: *
T0
q
 gradients_1/concat_14_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_14_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"
   
   
�
'gradients_1/concat_14_grad/ConcatOffsetConcatOffsetgradients_1/concat_14_grad/mod gradients_1/concat_14_grad/Shape"gradients_1/concat_14_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_14_grad/SliceSlice3gradients_1/MatMul_15_grad/tuple/control_dependency'gradients_1/concat_14_grad/ConcatOffset gradients_1/concat_14_grad/Shape*
_output_shapes

:
*
Index0*
T0
�
"gradients_1/concat_14_grad/Slice_1Slice3gradients_1/MatMul_15_grad/tuple/control_dependency)gradients_1/concat_14_grad/ConcatOffset:1"gradients_1/concat_14_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_14_grad/tuple/group_depsNoOp!^gradients_1/concat_14_grad/Slice#^gradients_1/concat_14_grad/Slice_1
�
3gradients_1/concat_14_grad/tuple/control_dependencyIdentity gradients_1/concat_14_grad/Slice,^gradients_1/concat_14_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/concat_14_grad/Slice*
_output_shapes

:

�
5gradients_1/concat_14_grad/tuple/control_dependency_1Identity"gradients_1/concat_14_grad/Slice_1,^gradients_1/concat_14_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/concat_14_grad/Slice_1*
_output_shapes

:

*
T0
�
!gradients_1/Tanh_13_grad/TanhGradTanhGradTanh_135gradients_1/concat_14_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_14_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_14_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_14_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_14_grad/Shapegradients_1/add_14_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_14_grad/SumSum!gradients_1/Tanh_13_grad/TanhGrad-gradients_1/add_14_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients_1/add_14_grad/ReshapeReshapegradients_1/add_14_grad/Sumgradients_1/add_14_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_14_grad/Sum_1Sum!gradients_1/Tanh_13_grad/TanhGrad/gradients_1/add_14_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_14_grad/Reshape_1Reshapegradients_1/add_14_grad/Sum_1gradients_1/add_14_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0
v
(gradients_1/add_14_grad/tuple/group_depsNoOp ^gradients_1/add_14_grad/Reshape"^gradients_1/add_14_grad/Reshape_1
�
0gradients_1/add_14_grad/tuple/control_dependencyIdentitygradients_1/add_14_grad/Reshape)^gradients_1/add_14_grad/tuple/group_deps*2
_class(
&$loc:@gradients_1/add_14_grad/Reshape*
_output_shapes

:

*
T0
�
2gradients_1/add_14_grad/tuple/control_dependency_1Identity!gradients_1/add_14_grad/Reshape_1)^gradients_1/add_14_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_14_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_14_grad/MatMulMatMul0gradients_1/add_14_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_14_grad/MatMul_1MatMul	concat_130gradients_1/add_14_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_14_grad/tuple/group_depsNoOp"^gradients_1/MatMul_14_grad/MatMul$^gradients_1/MatMul_14_grad/MatMul_1
�
3gradients_1/MatMul_14_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_14_grad/MatMul,^gradients_1/MatMul_14_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_14_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_14_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_14_grad/MatMul_1,^gradients_1/MatMul_14_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_14_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_13_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_13_grad/modFloorModconcat_13/axisgradients_1/concat_13_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_13_grad/ShapeConst*
_output_shapes
:*
valueB"
      *
dtype0
s
"gradients_1/concat_13_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"
   
   
�
'gradients_1/concat_13_grad/ConcatOffsetConcatOffsetgradients_1/concat_13_grad/mod gradients_1/concat_13_grad/Shape"gradients_1/concat_13_grad/Shape_1* 
_output_shapes
::*
N
�
 gradients_1/concat_13_grad/SliceSlice3gradients_1/MatMul_14_grad/tuple/control_dependency'gradients_1/concat_13_grad/ConcatOffset gradients_1/concat_13_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_13_grad/Slice_1Slice3gradients_1/MatMul_14_grad/tuple/control_dependency)gradients_1/concat_13_grad/ConcatOffset:1"gradients_1/concat_13_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
{
+gradients_1/concat_13_grad/tuple/group_depsNoOp!^gradients_1/concat_13_grad/Slice#^gradients_1/concat_13_grad/Slice_1
�
3gradients_1/concat_13_grad/tuple/control_dependencyIdentity gradients_1/concat_13_grad/Slice,^gradients_1/concat_13_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients_1/concat_13_grad/Slice
�
5gradients_1/concat_13_grad/tuple/control_dependency_1Identity"gradients_1/concat_13_grad/Slice_1,^gradients_1/concat_13_grad/tuple/group_deps*
_output_shapes

:

*
T0*5
_class+
)'loc:@gradients_1/concat_13_grad/Slice_1
�
!gradients_1/Tanh_12_grad/TanhGradTanhGradTanh_125gradients_1/concat_13_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_13_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_13_grad/Shape_1Const*
_output_shapes
:*
valueB"   
   *
dtype0
�
-gradients_1/add_13_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_13_grad/Shapegradients_1/add_13_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_13_grad/SumSum!gradients_1/Tanh_12_grad/TanhGrad-gradients_1/add_13_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients_1/add_13_grad/ReshapeReshapegradients_1/add_13_grad/Sumgradients_1/add_13_grad/Shape*
_output_shapes

:

*
T0*
Tshape0
�
gradients_1/add_13_grad/Sum_1Sum!gradients_1/Tanh_12_grad/TanhGrad/gradients_1/add_13_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
!gradients_1/add_13_grad/Reshape_1Reshapegradients_1/add_13_grad/Sum_1gradients_1/add_13_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_13_grad/tuple/group_depsNoOp ^gradients_1/add_13_grad/Reshape"^gradients_1/add_13_grad/Reshape_1
�
0gradients_1/add_13_grad/tuple/control_dependencyIdentitygradients_1/add_13_grad/Reshape)^gradients_1/add_13_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients_1/add_13_grad/Reshape
�
2gradients_1/add_13_grad/tuple/control_dependency_1Identity!gradients_1/add_13_grad/Reshape_1)^gradients_1/add_13_grad/tuple/group_deps*
_output_shapes

:
*
T0*4
_class*
(&loc:@gradients_1/add_13_grad/Reshape_1
�
!gradients_1/MatMul_13_grad/MatMulMatMul0gradients_1/add_13_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_13_grad/MatMul_1MatMul	concat_120gradients_1/add_13_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_13_grad/tuple/group_depsNoOp"^gradients_1/MatMul_13_grad/MatMul$^gradients_1/MatMul_13_grad/MatMul_1
�
3gradients_1/MatMul_13_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_13_grad/MatMul,^gradients_1/MatMul_13_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_13_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_13_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_13_grad/MatMul_1,^gradients_1/MatMul_13_grad/tuple/group_deps*6
_class,
*(loc:@gradients_1/MatMul_13_grad/MatMul_1*
_output_shapes

:
*
T0
a
gradients_1/concat_12_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_12_grad/modFloorModconcat_12/axisgradients_1/concat_12_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_12_grad/ShapeConst*
_output_shapes
:*
valueB"
      *
dtype0
s
"gradients_1/concat_12_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_12_grad/ConcatOffsetConcatOffsetgradients_1/concat_12_grad/mod gradients_1/concat_12_grad/Shape"gradients_1/concat_12_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_12_grad/SliceSlice3gradients_1/MatMul_13_grad/tuple/control_dependency'gradients_1/concat_12_grad/ConcatOffset gradients_1/concat_12_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_12_grad/Slice_1Slice3gradients_1/MatMul_13_grad/tuple/control_dependency)gradients_1/concat_12_grad/ConcatOffset:1"gradients_1/concat_12_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_12_grad/tuple/group_depsNoOp!^gradients_1/concat_12_grad/Slice#^gradients_1/concat_12_grad/Slice_1
�
3gradients_1/concat_12_grad/tuple/control_dependencyIdentity gradients_1/concat_12_grad/Slice,^gradients_1/concat_12_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/concat_12_grad/Slice*
_output_shapes

:

�
5gradients_1/concat_12_grad/tuple/control_dependency_1Identity"gradients_1/concat_12_grad/Slice_1,^gradients_1/concat_12_grad/tuple/group_deps*
_output_shapes

:

*
T0*5
_class+
)'loc:@gradients_1/concat_12_grad/Slice_1
�
!gradients_1/Tanh_11_grad/TanhGradTanhGradTanh_115gradients_1/concat_12_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_12_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
   
   
p
gradients_1/add_12_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_12_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_12_grad/Shapegradients_1/add_12_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_12_grad/SumSum!gradients_1/Tanh_11_grad/TanhGrad-gradients_1/add_12_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients_1/add_12_grad/ReshapeReshapegradients_1/add_12_grad/Sumgradients_1/add_12_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_12_grad/Sum_1Sum!gradients_1/Tanh_11_grad/TanhGrad/gradients_1/add_12_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
!gradients_1/add_12_grad/Reshape_1Reshapegradients_1/add_12_grad/Sum_1gradients_1/add_12_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_12_grad/tuple/group_depsNoOp ^gradients_1/add_12_grad/Reshape"^gradients_1/add_12_grad/Reshape_1
�
0gradients_1/add_12_grad/tuple/control_dependencyIdentitygradients_1/add_12_grad/Reshape)^gradients_1/add_12_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_12_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_12_grad/tuple/control_dependency_1Identity!gradients_1/add_12_grad/Reshape_1)^gradients_1/add_12_grad/tuple/group_deps*
_output_shapes

:
*
T0*4
_class*
(&loc:@gradients_1/add_12_grad/Reshape_1
�
!gradients_1/MatMul_12_grad/MatMulMatMul0gradients_1/add_12_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_12_grad/MatMul_1MatMul	concat_110gradients_1/add_12_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
}
+gradients_1/MatMul_12_grad/tuple/group_depsNoOp"^gradients_1/MatMul_12_grad/MatMul$^gradients_1/MatMul_12_grad/MatMul_1
�
3gradients_1/MatMul_12_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_12_grad/MatMul,^gradients_1/MatMul_12_grad/tuple/group_deps*
_output_shapes

:
*
T0*4
_class*
(&loc:@gradients_1/MatMul_12_grad/MatMul
�
5gradients_1/MatMul_12_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_12_grad/MatMul_1,^gradients_1/MatMul_12_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_12_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_11_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_11_grad/modFloorModconcat_11/axisgradients_1/concat_11_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_11_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
s
"gradients_1/concat_11_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_11_grad/ConcatOffsetConcatOffsetgradients_1/concat_11_grad/mod gradients_1/concat_11_grad/Shape"gradients_1/concat_11_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_11_grad/SliceSlice3gradients_1/MatMul_12_grad/tuple/control_dependency'gradients_1/concat_11_grad/ConcatOffset gradients_1/concat_11_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_11_grad/Slice_1Slice3gradients_1/MatMul_12_grad/tuple/control_dependency)gradients_1/concat_11_grad/ConcatOffset:1"gradients_1/concat_11_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_11_grad/tuple/group_depsNoOp!^gradients_1/concat_11_grad/Slice#^gradients_1/concat_11_grad/Slice_1
�
3gradients_1/concat_11_grad/tuple/control_dependencyIdentity gradients_1/concat_11_grad/Slice,^gradients_1/concat_11_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/concat_11_grad/Slice*
_output_shapes

:

�
5gradients_1/concat_11_grad/tuple/control_dependency_1Identity"gradients_1/concat_11_grad/Slice_1,^gradients_1/concat_11_grad/tuple/group_deps*
_output_shapes

:

*
T0*5
_class+
)'loc:@gradients_1/concat_11_grad/Slice_1
�
!gradients_1/Tanh_10_grad/TanhGradTanhGradTanh_105gradients_1/concat_11_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_11_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_11_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_11_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_11_grad/Shapegradients_1/add_11_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_11_grad/SumSum!gradients_1/Tanh_10_grad/TanhGrad-gradients_1/add_11_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients_1/add_11_grad/ReshapeReshapegradients_1/add_11_grad/Sumgradients_1/add_11_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_11_grad/Sum_1Sum!gradients_1/Tanh_10_grad/TanhGrad/gradients_1/add_11_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_11_grad/Reshape_1Reshapegradients_1/add_11_grad/Sum_1gradients_1/add_11_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_11_grad/tuple/group_depsNoOp ^gradients_1/add_11_grad/Reshape"^gradients_1/add_11_grad/Reshape_1
�
0gradients_1/add_11_grad/tuple/control_dependencyIdentitygradients_1/add_11_grad/Reshape)^gradients_1/add_11_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_11_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_11_grad/tuple/control_dependency_1Identity!gradients_1/add_11_grad/Reshape_1)^gradients_1/add_11_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_11_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_11_grad/MatMulMatMul0gradients_1/add_11_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_11_grad/MatMul_1MatMul	concat_100gradients_1/add_11_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_11_grad/tuple/group_depsNoOp"^gradients_1/MatMul_11_grad/MatMul$^gradients_1/MatMul_11_grad/MatMul_1
�
3gradients_1/MatMul_11_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_11_grad/MatMul,^gradients_1/MatMul_11_grad/tuple/group_deps*
_output_shapes

:
*
T0*4
_class*
(&loc:@gradients_1/MatMul_11_grad/MatMul
�
5gradients_1/MatMul_11_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_11_grad/MatMul_1,^gradients_1/MatMul_11_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_11_grad/MatMul_1*
_output_shapes

:

�
gradients_1/AddNAddN2gradients_1/add_20_grad/tuple/control_dependency_12gradients_1/add_19_grad/tuple/control_dependency_12gradients_1/add_18_grad/tuple/control_dependency_12gradients_1/add_17_grad/tuple/control_dependency_12gradients_1/add_16_grad/tuple/control_dependency_12gradients_1/add_15_grad/tuple/control_dependency_12gradients_1/add_14_grad/tuple/control_dependency_12gradients_1/add_13_grad/tuple/control_dependency_12gradients_1/add_12_grad/tuple/control_dependency_12gradients_1/add_11_grad/tuple/control_dependency_1*
T0*4
_class*
(&loc:@gradients_1/add_20_grad/Reshape_1*
N
*
_output_shapes

:

�
gradients_1/AddN_1AddN5gradients_1/MatMul_20_grad/tuple/control_dependency_15gradients_1/MatMul_19_grad/tuple/control_dependency_15gradients_1/MatMul_18_grad/tuple/control_dependency_15gradients_1/MatMul_17_grad/tuple/control_dependency_15gradients_1/MatMul_16_grad/tuple/control_dependency_15gradients_1/MatMul_15_grad/tuple/control_dependency_15gradients_1/MatMul_14_grad/tuple/control_dependency_15gradients_1/MatMul_13_grad/tuple/control_dependency_15gradients_1/MatMul_12_grad/tuple/control_dependency_15gradients_1/MatMul_11_grad/tuple/control_dependency_1*6
_class,
*(loc:@gradients_1/MatMul_20_grad/MatMul_1*
N
*
_output_shapes

:
*
T0

beta1_power_1/initial_valueConst*
_output_shapes
: *
_class
loc:@Variable_4*
valueB
 *fff?*
dtype0
�
beta1_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable_4*
	container *
shape: 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: *
use_locking(
m
beta1_power_1/readIdentitybeta1_power_1*
T0*
_class
loc:@Variable_4*
_output_shapes
: 

beta2_power_1/initial_valueConst*
_class
loc:@Variable_4*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power_1
VariableV2*
shared_name *
_class
loc:@Variable_4*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: 
m
beta2_power_1/readIdentitybeta2_power_1*
T0*
_class
loc:@Variable_4*
_output_shapes
: 
�
!Variable_4/Adam/Initializer/zerosConst*
_class
loc:@Variable_4*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_4/Adam
VariableV2*
_class
loc:@Variable_4*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
y
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4*
_output_shapes

:

�
#Variable_4/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_4*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_4/Adam_1
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_4*
	container 
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:

}
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4*
_output_shapes

:

�
!Variable_5/Adam/Initializer/zerosConst*
_class
loc:@Variable_5*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_5/Adam
VariableV2*
_class
loc:@Variable_5*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_5
y
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes

:
*
T0*
_class
loc:@Variable_5
�
#Variable_5/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_5*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_5/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_5*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
}
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes

:

�
!Variable_6/Adam/Initializer/zerosConst*
_class
loc:@Variable_6*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_6/Adam
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
loc:@Variable_6
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:

y
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes

:

�
#Variable_6/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_6*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_6/Adam_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_6*
	container *
shape
:

�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
}
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes

:

�
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@Variable_7*
valueB*    
�
Variable_7/Adam
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes
:
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
#Variable_7/Adam_1/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@Variable_7*
valueB*    *
dtype0
�
Variable_7/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes
:
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes
:*
T0*
_class
loc:@Variable_7
Y
Adam_1/learning_rateConst*
valueB
 *��8*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
"Adam_1/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_1*
use_locking( *
T0*
_class
loc:@Variable_4*
use_nesterov( *
_output_shapes

:

�
"Adam_1/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN*
_class
loc:@Variable_5*
use_nesterov( *
_output_shapes

:
*
use_locking( *
T0
�
"Adam_1/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon5gradients_1/MatMul_21_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:
*
use_locking( *
T0*
_class
loc:@Variable_6
�
"Adam_1/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon2gradients_1/add_21_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_7*
use_nesterov( *
_output_shapes
:*
use_locking( 
�

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1#^Adam_1/update_Variable_4/ApplyAdam#^Adam_1/update_Variable_5/ApplyAdam#^Adam_1/update_Variable_6/ApplyAdam#^Adam_1/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable_4*
_output_shapes
: 
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: *
use_locking( 
�
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2#^Adam_1/update_Variable_4/ApplyAdam#^Adam_1/update_Variable_5/ApplyAdam#^Adam_1/update_Variable_6/ApplyAdam#^Adam_1/update_Variable_7/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable_4
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: *
use_locking( 
�
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1#^Adam_1/update_Variable_4/ApplyAdam#^Adam_1/update_Variable_5/ApplyAdam#^Adam_1/update_Variable_6/ApplyAdam#^Adam_1/update_Variable_7/ApplyAdam
�
init_1NoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_6/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_7/Assign^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign"Ane�3     �:J�	8���	��AJ��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

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
1
Square
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.10.02v1.10.0-0-g656e7a2b34��
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
Placeholder_2Placeholder*
shape
:

*
dtype0*
_output_shapes

:


�
Variable/initial_valueConst*�
value�B�
"���P?��T?i��>�o�>�8�>�b?M�?��>f��>!*?�zw?�?�|C?w�<.��>U��>;`�=��-?r4Y?�U?m-?�{4?=q?��?�K�>�n�>�sK>�j+?�nN>�e ?&Y?��X?���>�=�=�M?��`?ߙC>9�1?32?�l�>8�?��j?8�a?!�<?^+8?���=0l)>9W�>���>�UU>>|��>��>Im�>o�?�Y?J�C=��I>��G?5ԃ>�{u?��==}�k?H>-7�>#?sxp>Y�
?��<�Ђ>>�jl?}�=>��>.�?E� ?��> ??p�%?���>1�>|:M?��?$�)>;#D>���>��N?�A?,E|?.uq?�R�>�%?��.?��>Y:"?�J�>�� :�)?xs?A�8?("@?��@?8�>��>��?���>��O>I� ?}��>t�2?*
dtype0*
_output_shapes

:

|
Variable
VariableV2*
_output_shapes

:
*
	container *
shape
:
*
shared_name *
dtype0
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
Variable/readIdentityVariable*
_output_shapes

:
*
T0*
_class
loc:@Variable
�
Variable_1/initial_valueConst*A
value8B6
"(                                        *
dtype0*
_output_shapes

:

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
_class
loc:@Variable_1*
_output_shapes

:
*
T0
�
Variable_2/initial_valueConst*A
value8B6
"([??�֕>�n?���>o�U>��<=�`>L�<?��?w�u>*
dtype0*
_output_shapes

:

~

Variable_2
VariableV2*
_output_shapes

:
*
	container *
shape
:
*
shared_name *
dtype0
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
Variable_3/initial_valueConst*
_output_shapes
:*
valueB*    *
dtype0
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
Variable_3Variable_3/initial_value*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:
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
ReshapeReshapeunstackReshape/shape*
_output_shapes

:
*
T0*
Tshape0
M
concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
u
concatConcatV2ReshapePlaceholder_2concat/axis*
T0*
N*
_output_shapes

:
*

Tidx0
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
addAddMatMulVariable_1/read*
_output_shapes

:

*
T0
:
TanhTanhadd*
T0*
_output_shapes

:


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
T0*
N*
_output_shapes

:
*

Tidx0
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
Tanh_1Tanhadd_1*
T0*
_output_shapes

:


`
Reshape_2/shapeConst*
dtype0*
_output_shapes
:*
valueB"
      
g
	Reshape_2Reshape	unstack:2Reshape_2/shape*
_output_shapes

:
*
T0*
Tshape0
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
MatMul_2MatMulconcat_2Variable/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
P
add_2AddMatMul_2Variable_1/read*
T0*
_output_shapes

:


>
Tanh_2Tanhadd_2*
T0*
_output_shapes

:


`
Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"
      
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
MatMul_3MatMulconcat_3Variable/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
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
Reshape_4/shapeConst*
_output_shapes
:*
valueB"
      *
dtype0
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
concat_4ConcatV2	Reshape_4Tanh_3concat_4/axis*
T0*
N*
_output_shapes

:
*

Tidx0
z
MatMul_4MatMulconcat_4Variable/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
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
N*
_output_shapes

:
*

Tidx0*
T0
z
MatMul_5MatMulconcat_5Variable/read*
transpose_b( *
T0*
_output_shapes

:

*
transpose_a( 
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
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_6Reshape	unstack:6Reshape_6/shape*
T0*
Tshape0*
_output_shapes

:

O
concat_6/axisConst*
value	B :*
dtype0*
_output_shapes
: 
t
concat_6ConcatV2	Reshape_6Tanh_5concat_6/axis*
T0*
N*
_output_shapes

:
*

Tidx0
z
MatMul_6MatMulconcat_6Variable/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
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
	Reshape_7Reshape	unstack:7Reshape_7/shape*
_output_shapes

:
*
T0*
Tshape0
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
MatMul_7MatMulconcat_7Variable/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
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
	Reshape_8Reshape	unstack:8Reshape_8/shape*
_output_shapes

:
*
T0*
Tshape0
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
MatMul_8MatMulconcat_8Variable/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
P
add_8AddMatMul_8Variable_1/read*
_output_shapes

:

*
T0
>
Tanh_8Tanhadd_8*
T0*
_output_shapes

:


`
Reshape_9/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
	Reshape_9Reshape	unstack:9Reshape_9/shape*
_output_shapes

:
*
T0*
Tshape0
O
concat_9/axisConst*
dtype0*
_output_shapes
: *
value	B :
t
concat_9ConcatV2	Reshape_9Tanh_8concat_9/axis*
_output_shapes

:
*

Tidx0*
T0*
N
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
add_9AddMatMul_9Variable_1/read*
T0*
_output_shapes

:


>
Tanh_9Tanhadd_9*
_output_shapes

:

*
T0
{
	MatMul_10MatMulTanh_9Variable_2/read*
_output_shapes

:
*
transpose_a( *
transpose_b( *
T0
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
SquareSquareSub*
T0*
_output_shapes

:

`
gradients/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
w
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes

:

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
gradients/Square_grad/Mul_1Mulgradients/Fillgradients/Square_grad/Mul*
_output_shapes

:
*
T0
e
gradients/Sub_grad/ShapeShapePlaceholder_1*
out_type0*
_output_shapes
:*
T0
k
gradients/Sub_grad/Shape_1Const*
_output_shapes
:*
valueB"
      *
dtype0
�
(gradients/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Sub_grad/Shapegradients/Sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/Sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Sub_grad/ReshapeReshapegradients/Sub_grad/Sumgradients/Sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/Sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/Sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Z
gradients/Sub_grad/NegNeggradients/Sub_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/Sub_grad/Reshape_1Reshapegradients/Sub_grad/Neggradients/Sub_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
g
#gradients/Sub_grad/tuple/group_depsNoOp^gradients/Sub_grad/Reshape^gradients/Sub_grad/Reshape_1
�
+gradients/Sub_grad/tuple/control_dependencyIdentitygradients/Sub_grad/Reshape$^gradients/Sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Sub_grad/Reshape*'
_output_shapes
:���������
�
-gradients/Sub_grad/tuple/control_dependency_1Identitygradients/Sub_grad/Reshape_1$^gradients/Sub_grad/tuple/group_deps*
_output_shapes

:
*
T0*/
_class%
#!loc:@gradients/Sub_grad/Reshape_1
l
gradients/add_10_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
gradients/add_10_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
+gradients/add_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_10_grad/Shapegradients/add_10_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_10_grad/SumSum-gradients/Sub_grad/tuple/control_dependency_1+gradients/add_10_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
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
gradients/add_10_grad/Reshape_1Reshapegradients/add_10_grad/Sum_1gradients/add_10_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
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
0gradients/add_10_grad/tuple/control_dependency_1Identitygradients/add_10_grad/Reshape_1'^gradients/add_10_grad/tuple/group_deps*2
_class(
&$loc:@gradients/add_10_grad/Reshape_1*
_output_shapes
:*
T0
�
gradients/MatMul_10_grad/MatMulMatMul.gradients/add_10_grad/tuple/control_dependencyVariable_2/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b(
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
1gradients/MatMul_10_grad/tuple/control_dependencyIdentitygradients/MatMul_10_grad/MatMul*^gradients/MatMul_10_grad/tuple/group_deps*2
_class(
&$loc:@gradients/MatMul_10_grad/MatMul*
_output_shapes

:

*
T0
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
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_9_grad/Shapegradients/add_9_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_9_grad/SumSumgradients/Tanh_9_grad/TanhGrad*gradients/add_9_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients/add_9_grad/ReshapeReshapegradients/add_9_grad/Sumgradients/add_9_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_9_grad/Sum_1Sumgradients/Tanh_9_grad/TanhGrad,gradients/add_9_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
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
/gradients/add_9_grad/tuple/control_dependency_1Identitygradients/add_9_grad/Reshape_1&^gradients/add_9_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/add_9_grad/Reshape_1
�
gradients/MatMul_9_grad/MatMulMatMul-gradients/add_9_grad/tuple/control_dependencyVariable/read*
transpose_b(*
T0*
_output_shapes

:
*
transpose_a( 
�
 gradients/MatMul_9_grad/MatMul_1MatMulconcat_9-gradients/add_9_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
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
dtype0*
_output_shapes
:*
valueB"
      
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
Index0*
T0*
_output_shapes

:

�
gradients/concat_9_grad/Slice_1Slice0gradients/MatMul_9_grad/tuple/control_dependency&gradients/concat_9_grad/ConcatOffset:1gradients/concat_9_grad/Shape_1*
Index0*
T0*
_output_shapes

:


r
(gradients/concat_9_grad/tuple/group_depsNoOp^gradients/concat_9_grad/Slice ^gradients/concat_9_grad/Slice_1
�
0gradients/concat_9_grad/tuple/control_dependencyIdentitygradients/concat_9_grad/Slice)^gradients/concat_9_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_9_grad/Slice*
_output_shapes

:

�
2gradients/concat_9_grad/tuple/control_dependency_1Identitygradients/concat_9_grad/Slice_1)^gradients/concat_9_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients/concat_9_grad/Slice_1
�
gradients/Tanh_8_grad/TanhGradTanhGradTanh_82gradients/concat_9_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
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
dtype0*
_output_shapes
:*
valueB"   
   
�
*gradients/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_8_grad/Shapegradients/add_8_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_8_grad/SumSumgradients/Tanh_8_grad/TanhGrad*gradients/add_8_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients/add_8_grad/ReshapeReshapegradients/add_8_grad/Sumgradients/add_8_grad/Shape*
_output_shapes

:

*
T0*
Tshape0
�
gradients/add_8_grad/Sum_1Sumgradients/Tanh_8_grad/TanhGrad,gradients/add_8_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients/add_8_grad/Reshape_1Reshapegradients/add_8_grad/Sum_1gradients/add_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

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
/gradients/add_8_grad/tuple/control_dependency_1Identitygradients/add_8_grad/Reshape_1&^gradients/add_8_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/add_8_grad/Reshape_1
�
gradients/MatMul_8_grad/MatMulMatMul-gradients/add_8_grad/tuple/control_dependencyVariable/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
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
gradients/concat_8_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0
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
_output_shapes

:

*
Index0*
T0
r
(gradients/concat_8_grad/tuple/group_depsNoOp^gradients/concat_8_grad/Slice ^gradients/concat_8_grad/Slice_1
�
0gradients/concat_8_grad/tuple/control_dependencyIdentitygradients/concat_8_grad/Slice)^gradients/concat_8_grad/tuple/group_deps*0
_class&
$"loc:@gradients/concat_8_grad/Slice*
_output_shapes

:
*
T0
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
gradients/add_7_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
m
gradients/add_7_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
*gradients/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_7_grad/Shapegradients/add_7_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_7_grad/SumSumgradients/Tanh_7_grad/TanhGrad*gradients/add_7_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients/add_7_grad/ReshapeReshapegradients/add_7_grad/Sumgradients/add_7_grad/Shape*
Tshape0*
_output_shapes

:

*
T0
�
gradients/add_7_grad/Sum_1Sumgradients/Tanh_7_grad/TanhGrad,gradients/add_7_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
gradients/add_7_grad/Reshape_1Reshapegradients/add_7_grad/Sum_1gradients/add_7_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

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
/gradients/add_7_grad/tuple/control_dependency_1Identitygradients/add_7_grad/Reshape_1&^gradients/add_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_7_grad/Reshape_1*
_output_shapes

:

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
0gradients/MatMul_7_grad/tuple/control_dependencyIdentitygradients/MatMul_7_grad/MatMul)^gradients/MatMul_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_7_grad/MatMul*
_output_shapes

:

�
2gradients/MatMul_7_grad/tuple/control_dependency_1Identity gradients/MatMul_7_grad/MatMul_1)^gradients/MatMul_7_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_7_grad/MatMul_1*
_output_shapes

:

^
gradients/concat_7_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0
u
gradients/concat_7_grad/modFloorModconcat_7/axisgradients/concat_7_grad/Rank*
T0*
_output_shapes
: 
n
gradients/concat_7_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
p
gradients/concat_7_grad/Shape_1Const*
_output_shapes
:*
valueB"
   
   *
dtype0
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
_output_shapes

:

*
Index0*
T0
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
2gradients/concat_7_grad/tuple/control_dependency_1Identitygradients/concat_7_grad/Slice_1)^gradients/concat_7_grad/tuple/group_deps*2
_class(
&$loc:@gradients/concat_7_grad/Slice_1*
_output_shapes

:

*
T0
�
gradients/Tanh_6_grad/TanhGradTanhGradTanh_62gradients/concat_7_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
k
gradients/add_6_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
   
   
m
gradients/add_6_grad/Shape_1Const*
_output_shapes
:*
valueB"   
   *
dtype0
�
*gradients/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_6_grad/Shapegradients/add_6_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_6_grad/SumSumgradients/Tanh_6_grad/TanhGrad*gradients/add_6_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients/add_6_grad/ReshapeReshapegradients/add_6_grad/Sumgradients/add_6_grad/Shape*
_output_shapes

:

*
T0*
Tshape0
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
-gradients/add_6_grad/tuple/control_dependencyIdentitygradients/add_6_grad/Reshape&^gradients/add_6_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_6_grad/Reshape*
_output_shapes

:

*
T0
�
/gradients/add_6_grad/tuple/control_dependency_1Identitygradients/add_6_grad/Reshape_1&^gradients/add_6_grad/tuple/group_deps*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/add_6_grad/Reshape_1
�
gradients/MatMul_6_grad/MatMulMatMul-gradients/add_6_grad/tuple/control_dependencyVariable/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_6_grad/MatMul_1MatMulconcat_6-gradients/add_6_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
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
gradients/concat_6_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_6_grad/modFloorModconcat_6/axisgradients/concat_6_grad/Rank*
T0*
_output_shapes
: 
n
gradients/concat_6_grad/ShapeConst*
_output_shapes
:*
valueB"
      *
dtype0
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
_output_shapes

:
*
Index0*
T0
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
2gradients/concat_6_grad/tuple/control_dependency_1Identitygradients/concat_6_grad/Slice_1)^gradients/concat_6_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients/concat_6_grad/Slice_1
�
gradients/Tanh_5_grad/TanhGradTanhGradTanh_52gradients/concat_6_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


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
gradients/add_5_grad/Sum_1Sumgradients/Tanh_5_grad/TanhGrad,gradients/add_5_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients/add_5_grad/Reshape_1Reshapegradients/add_5_grad/Sum_1gradients/add_5_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

m
%gradients/add_5_grad/tuple/group_depsNoOp^gradients/add_5_grad/Reshape^gradients/add_5_grad/Reshape_1
�
-gradients/add_5_grad/tuple/control_dependencyIdentitygradients/add_5_grad/Reshape&^gradients/add_5_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_5_grad/Reshape*
_output_shapes

:

*
T0
�
/gradients/add_5_grad/tuple/control_dependency_1Identitygradients/add_5_grad/Reshape_1&^gradients/add_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_5_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_5_grad/MatMulMatMul-gradients/add_5_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_5_grad/MatMul_1MatMulconcat_5-gradients/add_5_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_5_grad/tuple/group_depsNoOp^gradients/MatMul_5_grad/MatMul!^gradients/MatMul_5_grad/MatMul_1
�
0gradients/MatMul_5_grad/tuple/control_dependencyIdentitygradients/MatMul_5_grad/MatMul)^gradients/MatMul_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_5_grad/MatMul*
_output_shapes

:
*
T0
�
2gradients/MatMul_5_grad/tuple/control_dependency_1Identity gradients/MatMul_5_grad/MatMul_1)^gradients/MatMul_5_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_5_grad/MatMul_1*
_output_shapes

:

^
gradients/concat_5_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0
u
gradients/concat_5_grad/modFloorModconcat_5/axisgradients/concat_5_grad/Rank*
T0*
_output_shapes
: 
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
Index0*
T0*
_output_shapes

:


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
gradients/add_4_grad/SumSumgradients/Tanh_4_grad/TanhGrad*gradients/add_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
_output_shapes

:

*
T0*
Tshape0
�
gradients/add_4_grad/Sum_1Sumgradients/Tanh_4_grad/TanhGrad,gradients/add_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
�
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_4_grad/Reshape*
_output_shapes

:


�
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*
_output_shapes

:

�
gradients/MatMul_4_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyVariable/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
 gradients/MatMul_4_grad/MatMul_1MatMulconcat_4-gradients/add_4_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
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
2gradients/MatMul_4_grad/tuple/control_dependency_1Identity gradients/MatMul_4_grad/MatMul_1)^gradients/MatMul_4_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_4_grad/MatMul_1*
_output_shapes

:
*
T0
^
gradients/concat_4_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_4_grad/modFloorModconcat_4/axisgradients/concat_4_grad/Rank*
_output_shapes
: *
T0
n
gradients/concat_4_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
p
gradients/concat_4_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_4_grad/ConcatOffsetConcatOffsetgradients/concat_4_grad/modgradients/concat_4_grad/Shapegradients/concat_4_grad/Shape_1*
N* 
_output_shapes
::
�
gradients/concat_4_grad/SliceSlice0gradients/MatMul_4_grad/tuple/control_dependency$gradients/concat_4_grad/ConcatOffsetgradients/concat_4_grad/Shape*
_output_shapes

:
*
Index0*
T0
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
0gradients/concat_4_grad/tuple/control_dependencyIdentitygradients/concat_4_grad/Slice)^gradients/concat_4_grad/tuple/group_deps*
_output_shapes

:
*
T0*0
_class&
$"loc:@gradients/concat_4_grad/Slice
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
dtype0*
_output_shapes
:*
valueB"   
   
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_3_grad/SumSumgradients/Tanh_3_grad/TanhGrad*gradients/add_3_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_3_grad/Sum_1Sumgradients/Tanh_3_grad/TanhGrad,gradients/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

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
 gradients/MatMul_3_grad/MatMul_1MatMulconcat_3-gradients/add_3_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
�
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*
_output_shapes

:
*
T0
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
gradients/concat_3_grad/modFloorModconcat_3/axisgradients/concat_3_grad/Rank*
T0*
_output_shapes
: 
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
0gradients/concat_3_grad/tuple/control_dependencyIdentitygradients/concat_3_grad/Slice)^gradients/concat_3_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_3_grad/Slice*
_output_shapes

:

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
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_2_grad/SumSumgradients/Tanh_2_grad/TanhGrad*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0*
Tshape0*
_output_shapes

:


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
Tshape0*
_output_shapes

:
*
T0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
_output_shapes

:


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
 gradients/MatMul_2_grad/MatMul_1MatMulconcat_2-gradients/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
�
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*
_output_shapes

:
*
T0
�
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1*
_output_shapes

:
*
T0
^
gradients/concat_2_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
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
gradients/concat_2_grad/Shape_1Const*
_output_shapes
:*
valueB"
   
   *
dtype0
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
_output_shapes

:

*
Index0*
T0
r
(gradients/concat_2_grad/tuple/group_depsNoOp^gradients/concat_2_grad/Slice ^gradients/concat_2_grad/Slice_1
�
0gradients/concat_2_grad/tuple/control_dependencyIdentitygradients/concat_2_grad/Slice)^gradients/concat_2_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_2_grad/Slice*
_output_shapes

:

�
2gradients/concat_2_grad/tuple/control_dependency_1Identitygradients/concat_2_grad/Slice_1)^gradients/concat_2_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients/concat_2_grad/Slice_1
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
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Tanh_1_grad/TanhGrad*gradients/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_1_grad/Sum_1Sumgradients/Tanh_1_grad/TanhGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

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
 gradients/MatMul_1_grad/MatMul_1MatMulconcat_1-gradients/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
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
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:
*
T0
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
dtype0*
_output_shapes
:*
valueB"
      
p
gradients/concat_1_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/modgradients/concat_1_grad/Shapegradients/concat_1_grad/Shape_1*
N* 
_output_shapes
::
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
2gradients/concat_1_grad/tuple/control_dependency_1Identitygradients/concat_1_grad/Slice_1)^gradients/concat_1_grad/tuple/group_deps*2
_class(
&$loc:@gradients/concat_1_grad/Slice_1*
_output_shapes

:

*
T0
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
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Tanh_grad/TanhGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients/add_grad/Sum_1Sumgradients/Tanh_grad/TanhGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

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
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
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
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
_output_shapes

:

�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:

�
gradients/AddNAddN/gradients/add_9_grad/tuple/control_dependency_1/gradients/add_8_grad/tuple/control_dependency_1/gradients/add_7_grad/tuple/control_dependency_1/gradients/add_6_grad/tuple/control_dependency_1/gradients/add_5_grad/tuple/control_dependency_1/gradients/add_4_grad/tuple/control_dependency_1/gradients/add_3_grad/tuple/control_dependency_1/gradients/add_2_grad/tuple/control_dependency_1/gradients/add_1_grad/tuple/control_dependency_1-gradients/add_grad/tuple/control_dependency_1*
_output_shapes

:
*
T0*1
_class'
%#loc:@gradients/add_9_grad/Reshape_1*
N

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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
_output_shapes
: *
T0
{
beta2_power/initial_valueConst*
_output_shapes
: *
_class
loc:@Variable*
valueB
 *w�?*
dtype0
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*
_class
loc:@Variable
�
Variable/Adam/Initializer/zerosConst*
_class
loc:@Variable*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable/Adam
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
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
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable*
	container 
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:

w
Variable/Adam_1/readIdentityVariable/Adam_1*
_output_shapes

:
*
T0*
_class
loc:@Variable
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
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_1*
	container 
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
dtype0*
_output_shapes

:
*
_class
loc:@Variable_1*
valueB
*    
�
Variable_1/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_1*
	container *
shape
:
*
dtype0*
_output_shapes

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
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes

:
*
T0*
_class
loc:@Variable_1
�
!Variable_2/Adam/Initializer/zerosConst*
_output_shapes

:
*
_class
loc:@Variable_2*
valueB
*    *
dtype0
�
Variable_2/Adam
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_2*
	container *
shape
:

�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:

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
VariableV2*
shared_name *
_class
loc:@Variable_2*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:
*
use_locking(
}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
_output_shapes

:
*
T0
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
VariableV2*
shared_name *
_class
loc:@Variable_3*
	container *
shape:*
dtype0*
_output_shapes
:
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:*
use_locking(
u
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes
:
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
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes
:*
T0*
_class
loc:@Variable_3
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *��8
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
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
Adam/beta2Adam/epsilongradients/AddN_1*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( *
_output_shapes

:

�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( *
_output_shapes

:

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
Adam/beta2Adam/epsilon0gradients/add_10_grad/tuple/control_dependency_1*
_class
loc:@Variable_3*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable
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
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam*
_class
loc:@Variable*
_output_shapes
: *
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking( 
�
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam
�
initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^beta1_power/Assign^beta2_power/Assign
p
Placeholder_3Placeholder*
dtype0*'
_output_shapes
:���������
*
shape:���������

p
Placeholder_4Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
^
Placeholder_5Placeholder*
shape
:

*
dtype0*
_output_shapes

:


�
Variable_4/initial_valueConst*
dtype0*
_output_shapes

:
*�
value�B�
"�`p�>>?Ջ�>���>ĕ">~�"?Xa?���>��=i��>�>�*>��>>X/\?�I?4?�a�>)N�=�lb?5�>?�_>�?��F?nd�>3�?4t�>�ɻ>�G=l?*�?ب�={w?��=e�>�Lk<�/�=�w?.�>�>�9?��1=�>��U?���>C��>1��=�%�>bj?�Z?�x'>c�->bA?y�:?|P.?pq�>%@>.`?��c?H& >
�>��>�B?�W?�k8?->���>��?�c?��}?�>��h?>e?��t?�?!V?*�>��>�v?��S>fcw?O|>�J?=��>��>�?0=�
?�K=[O+?��>~?vv�>ꨁ>Ԡ�>]��>~�P?>�1?2�>��W>�O?�w�>'��>
�(?���>��F?n�>pz?41�>�M?�|?
~

Variable_4
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
Variable_4/AssignAssign
Variable_4Variable_4/initial_value*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:
*
use_locking(
o
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:

�
Variable_5/initial_valueConst*
dtype0*
_output_shapes

:
*A
value8B6
"(                                        
~

Variable_5
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
Variable_5/AssignAssign
Variable_5Variable_5/initial_value*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes

:

o
Variable_5/readIdentity
Variable_5*
_output_shapes

:
*
T0*
_class
loc:@Variable_5
�
Variable_6/initial_valueConst*A
value8B6
"(?	+?�+?��?�1?��i?t�A>��n?�4&?8�?67?*
dtype0*
_output_shapes

:

~

Variable_6
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
Variable_6/AssignAssign
Variable_6Variable_6/initial_value*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:

o
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes

:

e
Variable_7/initial_valueConst*
dtype0*
_output_shapes
:*
valueB*    
v

Variable_7
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
Variable_7/AssignAssign
Variable_7Variable_7/initial_value*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
	unstack_1UnpackPlaceholder_3*	
num
*
T0*

axis*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������
a
Reshape_10/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
i

Reshape_10Reshape	unstack_1Reshape_10/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_10/axisConst*
value	B :*
dtype0*
_output_shapes
: 
~
	concat_10ConcatV2
Reshape_10Placeholder_5concat_10/axis*
T0*
N*
_output_shapes

:
*

Tidx0
~
	MatMul_11MatMul	concat_10Variable_4/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
R
add_11Add	MatMul_11Variable_5/read*
T0*
_output_shapes

:


@
Tanh_10Tanhadd_11*
T0*
_output_shapes

:


a
Reshape_11/shapeConst*
_output_shapes
:*
valueB"
      *
dtype0
k

Reshape_11Reshapeunstack_1:1Reshape_11/shape*
_output_shapes

:
*
T0*
Tshape0
P
concat_11/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_11ConcatV2
Reshape_11Tanh_10concat_11/axis*

Tidx0*
T0*
N*
_output_shapes

:

~
	MatMul_12MatMul	concat_11Variable_4/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
R
add_12Add	MatMul_12Variable_5/read*
T0*
_output_shapes

:


@
Tanh_11Tanhadd_12*
T0*
_output_shapes

:


a
Reshape_12/shapeConst*
_output_shapes
:*
valueB"
      *
dtype0
k

Reshape_12Reshapeunstack_1:2Reshape_12/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_12/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_12ConcatV2
Reshape_12Tanh_11concat_12/axis*
_output_shapes

:
*

Tidx0*
T0*
N
~
	MatMul_13MatMul	concat_12Variable_4/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
R
add_13Add	MatMul_13Variable_5/read*
T0*
_output_shapes

:


@
Tanh_12Tanhadd_13*
_output_shapes

:

*
T0
a
Reshape_13/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_13Reshapeunstack_1:3Reshape_13/shape*
Tshape0*
_output_shapes

:
*
T0
P
concat_13/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_13ConcatV2
Reshape_13Tanh_12concat_13/axis*
_output_shapes

:
*

Tidx0*
T0*
N
~
	MatMul_14MatMul	concat_13Variable_4/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
R
add_14Add	MatMul_14Variable_5/read*
_output_shapes

:

*
T0
@
Tanh_13Tanhadd_14*
T0*
_output_shapes

:


a
Reshape_14/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_14Reshapeunstack_1:4Reshape_14/shape*
_output_shapes

:
*
T0*
Tshape0
P
concat_14/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_14ConcatV2
Reshape_14Tanh_13concat_14/axis*
T0*
N*
_output_shapes

:
*

Tidx0
~
	MatMul_15MatMul	concat_14Variable_4/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
R
add_15Add	MatMul_15Variable_5/read*
_output_shapes

:

*
T0
@
Tanh_14Tanhadd_15*
T0*
_output_shapes

:


a
Reshape_15/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_15Reshapeunstack_1:5Reshape_15/shape*
_output_shapes

:
*
T0*
Tshape0
P
concat_15/axisConst*
_output_shapes
: *
value	B :*
dtype0
x
	concat_15ConcatV2
Reshape_15Tanh_14concat_15/axis*
N*
_output_shapes

:
*

Tidx0*
T0
~
	MatMul_16MatMul	concat_15Variable_4/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
R
add_16Add	MatMul_16Variable_5/read*
T0*
_output_shapes

:


@
Tanh_15Tanhadd_16*
T0*
_output_shapes

:


a
Reshape_16/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_16Reshapeunstack_1:6Reshape_16/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_16/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_16ConcatV2
Reshape_16Tanh_15concat_16/axis*

Tidx0*
T0*
N*
_output_shapes

:

~
	MatMul_17MatMul	concat_16Variable_4/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
R
add_17Add	MatMul_17Variable_5/read*
T0*
_output_shapes

:


@
Tanh_16Tanhadd_17*
_output_shapes

:

*
T0
a
Reshape_17/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_17Reshapeunstack_1:7Reshape_17/shape*
_output_shapes

:
*
T0*
Tshape0
P
concat_17/axisConst*
value	B :*
dtype0*
_output_shapes
: 
x
	concat_17ConcatV2
Reshape_17Tanh_16concat_17/axis*

Tidx0*
T0*
N*
_output_shapes

:

~
	MatMul_18MatMul	concat_17Variable_4/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
R
add_18Add	MatMul_18Variable_5/read*
T0*
_output_shapes

:


@
Tanh_17Tanhadd_18*
T0*
_output_shapes

:


a
Reshape_18/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_18Reshapeunstack_1:8Reshape_18/shape*
T0*
Tshape0*
_output_shapes

:

P
concat_18/axisConst*
_output_shapes
: *
value	B :*
dtype0
x
	concat_18ConcatV2
Reshape_18Tanh_17concat_18/axis*
_output_shapes

:
*

Tidx0*
T0*
N
~
	MatMul_19MatMul	concat_18Variable_4/read*
_output_shapes

:

*
transpose_a( *
transpose_b( *
T0
R
add_19Add	MatMul_19Variable_5/read*
T0*
_output_shapes

:


@
Tanh_18Tanhadd_19*
_output_shapes

:

*
T0
a
Reshape_19/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
k

Reshape_19Reshapeunstack_1:9Reshape_19/shape*
_output_shapes

:
*
T0*
Tshape0
P
concat_19/axisConst*
dtype0*
_output_shapes
: *
value	B :
x
	concat_19ConcatV2
Reshape_19Tanh_18concat_19/axis*
T0*
N*
_output_shapes

:
*

Tidx0
~
	MatMul_20MatMul	concat_19Variable_4/read*
T0*
_output_shapes

:

*
transpose_a( *
transpose_b( 
R
add_20Add	MatMul_20Variable_5/read*
_output_shapes

:

*
T0
@
Tanh_19Tanhadd_20*
_output_shapes

:

*
T0
|
	MatMul_21MatMulTanh_19Variable_6/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b( 
R
add_21Add	MatMul_21Variable_7/read*
T0*
_output_shapes

:

L
Sub_1SubPlaceholder_4add_21*
T0*
_output_shapes

:

B
Square_1SquareSub_1*
T0*
_output_shapes

:

b
gradients_1/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
Z
gradients_1/grad_ys_0Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
}
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes

:

w
gradients_1/Square_1_grad/ConstConst^gradients_1/Fill*
valueB
 *   @*
dtype0*
_output_shapes
: 
u
gradients_1/Square_1_grad/MulMulSub_1gradients_1/Square_1_grad/Const*
T0*
_output_shapes

:

�
gradients_1/Square_1_grad/Mul_1Mulgradients_1/Fillgradients_1/Square_1_grad/Mul*
T0*
_output_shapes

:

i
gradients_1/Sub_1_grad/ShapeShapePlaceholder_4*
_output_shapes
:*
T0*
out_type0
o
gradients_1/Sub_1_grad/Shape_1Const*
valueB"
      *
dtype0*
_output_shapes
:
�
,gradients_1/Sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Sub_1_grad/Shapegradients_1/Sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/Sub_1_grad/SumSumgradients_1/Square_1_grad/Mul_1,gradients_1/Sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients_1/Sub_1_grad/ReshapeReshapegradients_1/Sub_1_grad/Sumgradients_1/Sub_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients_1/Sub_1_grad/Sum_1Sumgradients_1/Square_1_grad/Mul_1.gradients_1/Sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
b
gradients_1/Sub_1_grad/NegNeggradients_1/Sub_1_grad/Sum_1*
T0*
_output_shapes
:
�
 gradients_1/Sub_1_grad/Reshape_1Reshapegradients_1/Sub_1_grad/Neggradients_1/Sub_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

s
'gradients_1/Sub_1_grad/tuple/group_depsNoOp^gradients_1/Sub_1_grad/Reshape!^gradients_1/Sub_1_grad/Reshape_1
�
/gradients_1/Sub_1_grad/tuple/control_dependencyIdentitygradients_1/Sub_1_grad/Reshape(^gradients_1/Sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/Sub_1_grad/Reshape*'
_output_shapes
:���������
�
1gradients_1/Sub_1_grad/tuple/control_dependency_1Identity gradients_1/Sub_1_grad/Reshape_1(^gradients_1/Sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/Sub_1_grad/Reshape_1*
_output_shapes

:

n
gradients_1/add_21_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
i
gradients_1/add_21_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
-gradients_1/add_21_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_21_grad/Shapegradients_1/add_21_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_21_grad/SumSum1gradients_1/Sub_1_grad/tuple/control_dependency_1-gradients_1/add_21_grad/BroadcastGradientArgs*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
gradients_1/add_21_grad/ReshapeReshapegradients_1/add_21_grad/Sumgradients_1/add_21_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
gradients_1/add_21_grad/Sum_1Sum1gradients_1/Sub_1_grad/tuple/control_dependency_1/gradients_1/add_21_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
!gradients_1/add_21_grad/Reshape_1Reshapegradients_1/add_21_grad/Sum_1gradients_1/add_21_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
v
(gradients_1/add_21_grad/tuple/group_depsNoOp ^gradients_1/add_21_grad/Reshape"^gradients_1/add_21_grad/Reshape_1
�
0gradients_1/add_21_grad/tuple/control_dependencyIdentitygradients_1/add_21_grad/Reshape)^gradients_1/add_21_grad/tuple/group_deps*
_output_shapes

:
*
T0*2
_class(
&$loc:@gradients_1/add_21_grad/Reshape
�
2gradients_1/add_21_grad/tuple/control_dependency_1Identity!gradients_1/add_21_grad/Reshape_1)^gradients_1/add_21_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_21_grad/Reshape_1*
_output_shapes
:
�
!gradients_1/MatMul_21_grad/MatMulMatMul0gradients_1/add_21_grad/tuple/control_dependencyVariable_6/read*
_output_shapes

:

*
transpose_a( *
transpose_b(*
T0
�
#gradients_1/MatMul_21_grad/MatMul_1MatMulTanh_190gradients_1/add_21_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_21_grad/tuple/group_depsNoOp"^gradients_1/MatMul_21_grad/MatMul$^gradients_1/MatMul_21_grad/MatMul_1
�
3gradients_1/MatMul_21_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_21_grad/MatMul,^gradients_1/MatMul_21_grad/tuple/group_deps*
_output_shapes

:

*
T0*4
_class*
(&loc:@gradients_1/MatMul_21_grad/MatMul
�
5gradients_1/MatMul_21_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_21_grad/MatMul_1,^gradients_1/MatMul_21_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_21_grad/MatMul_1*
_output_shapes

:

�
!gradients_1/Tanh_19_grad/TanhGradTanhGradTanh_193gradients_1/MatMul_21_grad/tuple/control_dependency*
_output_shapes

:

*
T0
n
gradients_1/add_20_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_20_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_20_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_20_grad/Shapegradients_1/add_20_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_20_grad/SumSum!gradients_1/Tanh_19_grad/TanhGrad-gradients_1/add_20_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients_1/add_20_grad/ReshapeReshapegradients_1/add_20_grad/Sumgradients_1/add_20_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_20_grad/Sum_1Sum!gradients_1/Tanh_19_grad/TanhGrad/gradients_1/add_20_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
!gradients_1/add_20_grad/Reshape_1Reshapegradients_1/add_20_grad/Sum_1gradients_1/add_20_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_20_grad/tuple/group_depsNoOp ^gradients_1/add_20_grad/Reshape"^gradients_1/add_20_grad/Reshape_1
�
0gradients_1/add_20_grad/tuple/control_dependencyIdentitygradients_1/add_20_grad/Reshape)^gradients_1/add_20_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_20_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_20_grad/tuple/control_dependency_1Identity!gradients_1/add_20_grad/Reshape_1)^gradients_1/add_20_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/add_20_grad/Reshape_1*
_output_shapes

:
*
T0
�
!gradients_1/MatMul_20_grad/MatMulMatMul0gradients_1/add_20_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_20_grad/MatMul_1MatMul	concat_190gradients_1/add_20_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_20_grad/tuple/group_depsNoOp"^gradients_1/MatMul_20_grad/MatMul$^gradients_1/MatMul_20_grad/MatMul_1
�
3gradients_1/MatMul_20_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_20_grad/MatMul,^gradients_1/MatMul_20_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_20_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_20_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_20_grad/MatMul_1,^gradients_1/MatMul_20_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_20_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_19_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_19_grad/modFloorModconcat_19/axisgradients_1/concat_19_grad/Rank*
_output_shapes
: *
T0
q
 gradients_1/concat_19_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_19_grad/Shape_1Const*
_output_shapes
:*
valueB"
   
   *
dtype0
�
'gradients_1/concat_19_grad/ConcatOffsetConcatOffsetgradients_1/concat_19_grad/mod gradients_1/concat_19_grad/Shape"gradients_1/concat_19_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_19_grad/SliceSlice3gradients_1/MatMul_20_grad/tuple/control_dependency'gradients_1/concat_19_grad/ConcatOffset gradients_1/concat_19_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_19_grad/Slice_1Slice3gradients_1/MatMul_20_grad/tuple/control_dependency)gradients_1/concat_19_grad/ConcatOffset:1"gradients_1/concat_19_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_19_grad/tuple/group_depsNoOp!^gradients_1/concat_19_grad/Slice#^gradients_1/concat_19_grad/Slice_1
�
3gradients_1/concat_19_grad/tuple/control_dependencyIdentity gradients_1/concat_19_grad/Slice,^gradients_1/concat_19_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients_1/concat_19_grad/Slice
�
5gradients_1/concat_19_grad/tuple/control_dependency_1Identity"gradients_1/concat_19_grad/Slice_1,^gradients_1/concat_19_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/concat_19_grad/Slice_1*
_output_shapes

:

*
T0
�
!gradients_1/Tanh_18_grad/TanhGradTanhGradTanh_185gradients_1/concat_19_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_19_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_19_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_19_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_19_grad/Shapegradients_1/add_19_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_19_grad/SumSum!gradients_1/Tanh_18_grad/TanhGrad-gradients_1/add_19_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients_1/add_19_grad/ReshapeReshapegradients_1/add_19_grad/Sumgradients_1/add_19_grad/Shape*
_output_shapes

:

*
T0*
Tshape0
�
gradients_1/add_19_grad/Sum_1Sum!gradients_1/Tanh_18_grad/TanhGrad/gradients_1/add_19_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
!gradients_1/add_19_grad/Reshape_1Reshapegradients_1/add_19_grad/Sum_1gradients_1/add_19_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_19_grad/tuple/group_depsNoOp ^gradients_1/add_19_grad/Reshape"^gradients_1/add_19_grad/Reshape_1
�
0gradients_1/add_19_grad/tuple/control_dependencyIdentitygradients_1/add_19_grad/Reshape)^gradients_1/add_19_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_19_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_19_grad/tuple/control_dependency_1Identity!gradients_1/add_19_grad/Reshape_1)^gradients_1/add_19_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_19_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_19_grad/MatMulMatMul0gradients_1/add_19_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_19_grad/MatMul_1MatMul	concat_180gradients_1/add_19_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_19_grad/tuple/group_depsNoOp"^gradients_1/MatMul_19_grad/MatMul$^gradients_1/MatMul_19_grad/MatMul_1
�
3gradients_1/MatMul_19_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_19_grad/MatMul,^gradients_1/MatMul_19_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_19_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_19_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_19_grad/MatMul_1,^gradients_1/MatMul_19_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_19_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_18_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_18_grad/modFloorModconcat_18/axisgradients_1/concat_18_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_18_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_18_grad/Shape_1Const*
_output_shapes
:*
valueB"
   
   *
dtype0
�
'gradients_1/concat_18_grad/ConcatOffsetConcatOffsetgradients_1/concat_18_grad/mod gradients_1/concat_18_grad/Shape"gradients_1/concat_18_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_18_grad/SliceSlice3gradients_1/MatMul_19_grad/tuple/control_dependency'gradients_1/concat_18_grad/ConcatOffset gradients_1/concat_18_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_18_grad/Slice_1Slice3gradients_1/MatMul_19_grad/tuple/control_dependency)gradients_1/concat_18_grad/ConcatOffset:1"gradients_1/concat_18_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_18_grad/tuple/group_depsNoOp!^gradients_1/concat_18_grad/Slice#^gradients_1/concat_18_grad/Slice_1
�
3gradients_1/concat_18_grad/tuple/control_dependencyIdentity gradients_1/concat_18_grad/Slice,^gradients_1/concat_18_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/concat_18_grad/Slice*
_output_shapes

:
*
T0
�
5gradients_1/concat_18_grad/tuple/control_dependency_1Identity"gradients_1/concat_18_grad/Slice_1,^gradients_1/concat_18_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/concat_18_grad/Slice_1*
_output_shapes

:

*
T0
�
!gradients_1/Tanh_17_grad/TanhGradTanhGradTanh_175gradients_1/concat_18_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_18_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_18_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_18_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_18_grad/Shapegradients_1/add_18_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_18_grad/SumSum!gradients_1/Tanh_17_grad/TanhGrad-gradients_1/add_18_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:


�
gradients_1/add_18_grad/ReshapeReshapegradients_1/add_18_grad/Sumgradients_1/add_18_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_18_grad/Sum_1Sum!gradients_1/Tanh_17_grad/TanhGrad/gradients_1/add_18_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_18_grad/Reshape_1Reshapegradients_1/add_18_grad/Sum_1gradients_1/add_18_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
v
(gradients_1/add_18_grad/tuple/group_depsNoOp ^gradients_1/add_18_grad/Reshape"^gradients_1/add_18_grad/Reshape_1
�
0gradients_1/add_18_grad/tuple/control_dependencyIdentitygradients_1/add_18_grad/Reshape)^gradients_1/add_18_grad/tuple/group_deps*2
_class(
&$loc:@gradients_1/add_18_grad/Reshape*
_output_shapes

:

*
T0
�
2gradients_1/add_18_grad/tuple/control_dependency_1Identity!gradients_1/add_18_grad/Reshape_1)^gradients_1/add_18_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_18_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_18_grad/MatMulMatMul0gradients_1/add_18_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_18_grad/MatMul_1MatMul	concat_170gradients_1/add_18_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
}
+gradients_1/MatMul_18_grad/tuple/group_depsNoOp"^gradients_1/MatMul_18_grad/MatMul$^gradients_1/MatMul_18_grad/MatMul_1
�
3gradients_1/MatMul_18_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_18_grad/MatMul,^gradients_1/MatMul_18_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_18_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_18_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_18_grad/MatMul_1,^gradients_1/MatMul_18_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_18_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_17_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0
|
gradients_1/concat_17_grad/modFloorModconcat_17/axisgradients_1/concat_17_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_17_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_17_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_17_grad/ConcatOffsetConcatOffsetgradients_1/concat_17_grad/mod gradients_1/concat_17_grad/Shape"gradients_1/concat_17_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_17_grad/SliceSlice3gradients_1/MatMul_18_grad/tuple/control_dependency'gradients_1/concat_17_grad/ConcatOffset gradients_1/concat_17_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_17_grad/Slice_1Slice3gradients_1/MatMul_18_grad/tuple/control_dependency)gradients_1/concat_17_grad/ConcatOffset:1"gradients_1/concat_17_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
{
+gradients_1/concat_17_grad/tuple/group_depsNoOp!^gradients_1/concat_17_grad/Slice#^gradients_1/concat_17_grad/Slice_1
�
3gradients_1/concat_17_grad/tuple/control_dependencyIdentity gradients_1/concat_17_grad/Slice,^gradients_1/concat_17_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/concat_17_grad/Slice*
_output_shapes

:

�
5gradients_1/concat_17_grad/tuple/control_dependency_1Identity"gradients_1/concat_17_grad/Slice_1,^gradients_1/concat_17_grad/tuple/group_deps*
_output_shapes

:

*
T0*5
_class+
)'loc:@gradients_1/concat_17_grad/Slice_1
�
!gradients_1/Tanh_16_grad/TanhGradTanhGradTanh_165gradients_1/concat_17_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_17_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_17_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_17_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_17_grad/Shapegradients_1/add_17_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_17_grad/SumSum!gradients_1/Tanh_16_grad/TanhGrad-gradients_1/add_17_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients_1/add_17_grad/ReshapeReshapegradients_1/add_17_grad/Sumgradients_1/add_17_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_17_grad/Sum_1Sum!gradients_1/Tanh_16_grad/TanhGrad/gradients_1/add_17_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_17_grad/Reshape_1Reshapegradients_1/add_17_grad/Sum_1gradients_1/add_17_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
v
(gradients_1/add_17_grad/tuple/group_depsNoOp ^gradients_1/add_17_grad/Reshape"^gradients_1/add_17_grad/Reshape_1
�
0gradients_1/add_17_grad/tuple/control_dependencyIdentitygradients_1/add_17_grad/Reshape)^gradients_1/add_17_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients_1/add_17_grad/Reshape
�
2gradients_1/add_17_grad/tuple/control_dependency_1Identity!gradients_1/add_17_grad/Reshape_1)^gradients_1/add_17_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_17_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_17_grad/MatMulMatMul0gradients_1/add_17_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_17_grad/MatMul_1MatMul	concat_160gradients_1/add_17_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_17_grad/tuple/group_depsNoOp"^gradients_1/MatMul_17_grad/MatMul$^gradients_1/MatMul_17_grad/MatMul_1
�
3gradients_1/MatMul_17_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_17_grad/MatMul,^gradients_1/MatMul_17_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_17_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_17_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_17_grad/MatMul_1,^gradients_1/MatMul_17_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_17_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_16_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_16_grad/modFloorModconcat_16/axisgradients_1/concat_16_grad/Rank*
_output_shapes
: *
T0
q
 gradients_1/concat_16_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_16_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_16_grad/ConcatOffsetConcatOffsetgradients_1/concat_16_grad/mod gradients_1/concat_16_grad/Shape"gradients_1/concat_16_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_16_grad/SliceSlice3gradients_1/MatMul_17_grad/tuple/control_dependency'gradients_1/concat_16_grad/ConcatOffset gradients_1/concat_16_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_16_grad/Slice_1Slice3gradients_1/MatMul_17_grad/tuple/control_dependency)gradients_1/concat_16_grad/ConcatOffset:1"gradients_1/concat_16_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_16_grad/tuple/group_depsNoOp!^gradients_1/concat_16_grad/Slice#^gradients_1/concat_16_grad/Slice_1
�
3gradients_1/concat_16_grad/tuple/control_dependencyIdentity gradients_1/concat_16_grad/Slice,^gradients_1/concat_16_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/concat_16_grad/Slice*
_output_shapes

:

�
5gradients_1/concat_16_grad/tuple/control_dependency_1Identity"gradients_1/concat_16_grad/Slice_1,^gradients_1/concat_16_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/concat_16_grad/Slice_1*
_output_shapes

:


�
!gradients_1/Tanh_15_grad/TanhGradTanhGradTanh_155gradients_1/concat_16_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_16_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_16_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_16_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_16_grad/Shapegradients_1/add_16_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_16_grad/SumSum!gradients_1/Tanh_15_grad/TanhGrad-gradients_1/add_16_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients_1/add_16_grad/ReshapeReshapegradients_1/add_16_grad/Sumgradients_1/add_16_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_16_grad/Sum_1Sum!gradients_1/Tanh_15_grad/TanhGrad/gradients_1/add_16_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_16_grad/Reshape_1Reshapegradients_1/add_16_grad/Sum_1gradients_1/add_16_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_16_grad/tuple/group_depsNoOp ^gradients_1/add_16_grad/Reshape"^gradients_1/add_16_grad/Reshape_1
�
0gradients_1/add_16_grad/tuple/control_dependencyIdentitygradients_1/add_16_grad/Reshape)^gradients_1/add_16_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_16_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_16_grad/tuple/control_dependency_1Identity!gradients_1/add_16_grad/Reshape_1)^gradients_1/add_16_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/add_16_grad/Reshape_1*
_output_shapes

:
*
T0
�
!gradients_1/MatMul_16_grad/MatMulMatMul0gradients_1/add_16_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_16_grad/MatMul_1MatMul	concat_150gradients_1/add_16_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_16_grad/tuple/group_depsNoOp"^gradients_1/MatMul_16_grad/MatMul$^gradients_1/MatMul_16_grad/MatMul_1
�
3gradients_1/MatMul_16_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_16_grad/MatMul,^gradients_1/MatMul_16_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_16_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_16_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_16_grad/MatMul_1,^gradients_1/MatMul_16_grad/tuple/group_deps*
_output_shapes

:
*
T0*6
_class,
*(loc:@gradients_1/MatMul_16_grad/MatMul_1
a
gradients_1/concat_15_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_15_grad/modFloorModconcat_15/axisgradients_1/concat_15_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_15_grad/ShapeConst*
_output_shapes
:*
valueB"
      *
dtype0
s
"gradients_1/concat_15_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"
   
   
�
'gradients_1/concat_15_grad/ConcatOffsetConcatOffsetgradients_1/concat_15_grad/mod gradients_1/concat_15_grad/Shape"gradients_1/concat_15_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_15_grad/SliceSlice3gradients_1/MatMul_16_grad/tuple/control_dependency'gradients_1/concat_15_grad/ConcatOffset gradients_1/concat_15_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_15_grad/Slice_1Slice3gradients_1/MatMul_16_grad/tuple/control_dependency)gradients_1/concat_15_grad/ConcatOffset:1"gradients_1/concat_15_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_15_grad/tuple/group_depsNoOp!^gradients_1/concat_15_grad/Slice#^gradients_1/concat_15_grad/Slice_1
�
3gradients_1/concat_15_grad/tuple/control_dependencyIdentity gradients_1/concat_15_grad/Slice,^gradients_1/concat_15_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients_1/concat_15_grad/Slice
�
5gradients_1/concat_15_grad/tuple/control_dependency_1Identity"gradients_1/concat_15_grad/Slice_1,^gradients_1/concat_15_grad/tuple/group_deps*
_output_shapes

:

*
T0*5
_class+
)'loc:@gradients_1/concat_15_grad/Slice_1
�
!gradients_1/Tanh_14_grad/TanhGradTanhGradTanh_145gradients_1/concat_15_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_15_grad/ShapeConst*
_output_shapes
:*
valueB"
   
   *
dtype0
p
gradients_1/add_15_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_15_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_15_grad/Shapegradients_1/add_15_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_15_grad/SumSum!gradients_1/Tanh_14_grad/TanhGrad-gradients_1/add_15_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients_1/add_15_grad/ReshapeReshapegradients_1/add_15_grad/Sumgradients_1/add_15_grad/Shape*
_output_shapes

:

*
T0*
Tshape0
�
gradients_1/add_15_grad/Sum_1Sum!gradients_1/Tanh_14_grad/TanhGrad/gradients_1/add_15_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_15_grad/Reshape_1Reshapegradients_1/add_15_grad/Sum_1gradients_1/add_15_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_15_grad/tuple/group_depsNoOp ^gradients_1/add_15_grad/Reshape"^gradients_1/add_15_grad/Reshape_1
�
0gradients_1/add_15_grad/tuple/control_dependencyIdentitygradients_1/add_15_grad/Reshape)^gradients_1/add_15_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients_1/add_15_grad/Reshape
�
2gradients_1/add_15_grad/tuple/control_dependency_1Identity!gradients_1/add_15_grad/Reshape_1)^gradients_1/add_15_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_15_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_15_grad/MatMulMatMul0gradients_1/add_15_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_15_grad/MatMul_1MatMul	concat_140gradients_1/add_15_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
}
+gradients_1/MatMul_15_grad/tuple/group_depsNoOp"^gradients_1/MatMul_15_grad/MatMul$^gradients_1/MatMul_15_grad/MatMul_1
�
3gradients_1/MatMul_15_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_15_grad/MatMul,^gradients_1/MatMul_15_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_15_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_15_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_15_grad/MatMul_1,^gradients_1/MatMul_15_grad/tuple/group_deps*
_output_shapes

:
*
T0*6
_class,
*(loc:@gradients_1/MatMul_15_grad/MatMul_1
a
gradients_1/concat_14_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_14_grad/modFloorModconcat_14/axisgradients_1/concat_14_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_14_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
s
"gradients_1/concat_14_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_14_grad/ConcatOffsetConcatOffsetgradients_1/concat_14_grad/mod gradients_1/concat_14_grad/Shape"gradients_1/concat_14_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_14_grad/SliceSlice3gradients_1/MatMul_15_grad/tuple/control_dependency'gradients_1/concat_14_grad/ConcatOffset gradients_1/concat_14_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_14_grad/Slice_1Slice3gradients_1/MatMul_15_grad/tuple/control_dependency)gradients_1/concat_14_grad/ConcatOffset:1"gradients_1/concat_14_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
{
+gradients_1/concat_14_grad/tuple/group_depsNoOp!^gradients_1/concat_14_grad/Slice#^gradients_1/concat_14_grad/Slice_1
�
3gradients_1/concat_14_grad/tuple/control_dependencyIdentity gradients_1/concat_14_grad/Slice,^gradients_1/concat_14_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/concat_14_grad/Slice*
_output_shapes

:

�
5gradients_1/concat_14_grad/tuple/control_dependency_1Identity"gradients_1/concat_14_grad/Slice_1,^gradients_1/concat_14_grad/tuple/group_deps*
_output_shapes

:

*
T0*5
_class+
)'loc:@gradients_1/concat_14_grad/Slice_1
�
!gradients_1/Tanh_13_grad/TanhGradTanhGradTanh_135gradients_1/concat_14_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
n
gradients_1/add_14_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
   
   
p
gradients_1/add_14_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_14_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_14_grad/Shapegradients_1/add_14_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_14_grad/SumSum!gradients_1/Tanh_13_grad/TanhGrad-gradients_1/add_14_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients_1/add_14_grad/ReshapeReshapegradients_1/add_14_grad/Sumgradients_1/add_14_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_14_grad/Sum_1Sum!gradients_1/Tanh_13_grad/TanhGrad/gradients_1/add_14_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
*
	keep_dims( *

Tidx0
�
!gradients_1/add_14_grad/Reshape_1Reshapegradients_1/add_14_grad/Sum_1gradients_1/add_14_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_14_grad/tuple/group_depsNoOp ^gradients_1/add_14_grad/Reshape"^gradients_1/add_14_grad/Reshape_1
�
0gradients_1/add_14_grad/tuple/control_dependencyIdentitygradients_1/add_14_grad/Reshape)^gradients_1/add_14_grad/tuple/group_deps*
_output_shapes

:

*
T0*2
_class(
&$loc:@gradients_1/add_14_grad/Reshape
�
2gradients_1/add_14_grad/tuple/control_dependency_1Identity!gradients_1/add_14_grad/Reshape_1)^gradients_1/add_14_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/add_14_grad/Reshape_1*
_output_shapes

:
*
T0
�
!gradients_1/MatMul_14_grad/MatMulMatMul0gradients_1/add_14_grad/tuple/control_dependencyVariable_4/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
�
#gradients_1/MatMul_14_grad/MatMul_1MatMul	concat_130gradients_1/add_14_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
}
+gradients_1/MatMul_14_grad/tuple/group_depsNoOp"^gradients_1/MatMul_14_grad/MatMul$^gradients_1/MatMul_14_grad/MatMul_1
�
3gradients_1/MatMul_14_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_14_grad/MatMul,^gradients_1/MatMul_14_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_14_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_14_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_14_grad/MatMul_1,^gradients_1/MatMul_14_grad/tuple/group_deps*
_output_shapes

:
*
T0*6
_class,
*(loc:@gradients_1/MatMul_14_grad/MatMul_1
a
gradients_1/concat_13_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
|
gradients_1/concat_13_grad/modFloorModconcat_13/axisgradients_1/concat_13_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_13_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_13_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_13_grad/ConcatOffsetConcatOffsetgradients_1/concat_13_grad/mod gradients_1/concat_13_grad/Shape"gradients_1/concat_13_grad/Shape_1*
N* 
_output_shapes
::
�
 gradients_1/concat_13_grad/SliceSlice3gradients_1/MatMul_14_grad/tuple/control_dependency'gradients_1/concat_13_grad/ConcatOffset gradients_1/concat_13_grad/Shape*
Index0*
T0*
_output_shapes

:

�
"gradients_1/concat_13_grad/Slice_1Slice3gradients_1/MatMul_14_grad/tuple/control_dependency)gradients_1/concat_13_grad/ConcatOffset:1"gradients_1/concat_13_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
{
+gradients_1/concat_13_grad/tuple/group_depsNoOp!^gradients_1/concat_13_grad/Slice#^gradients_1/concat_13_grad/Slice_1
�
3gradients_1/concat_13_grad/tuple/control_dependencyIdentity gradients_1/concat_13_grad/Slice,^gradients_1/concat_13_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/concat_13_grad/Slice*
_output_shapes

:
*
T0
�
5gradients_1/concat_13_grad/tuple/control_dependency_1Identity"gradients_1/concat_13_grad/Slice_1,^gradients_1/concat_13_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/concat_13_grad/Slice_1*
_output_shapes

:


�
!gradients_1/Tanh_12_grad/TanhGradTanhGradTanh_125gradients_1/concat_13_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
n
gradients_1/add_13_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
   
   
p
gradients_1/add_13_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_13_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_13_grad/Shapegradients_1/add_13_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_13_grad/SumSum!gradients_1/Tanh_12_grad/TanhGrad-gradients_1/add_13_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients_1/add_13_grad/ReshapeReshapegradients_1/add_13_grad/Sumgradients_1/add_13_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_13_grad/Sum_1Sum!gradients_1/Tanh_12_grad/TanhGrad/gradients_1/add_13_grad/BroadcastGradientArgs:1*
_output_shapes
:
*
	keep_dims( *

Tidx0*
T0
�
!gradients_1/add_13_grad/Reshape_1Reshapegradients_1/add_13_grad/Sum_1gradients_1/add_13_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
v
(gradients_1/add_13_grad/tuple/group_depsNoOp ^gradients_1/add_13_grad/Reshape"^gradients_1/add_13_grad/Reshape_1
�
0gradients_1/add_13_grad/tuple/control_dependencyIdentitygradients_1/add_13_grad/Reshape)^gradients_1/add_13_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_13_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_13_grad/tuple/control_dependency_1Identity!gradients_1/add_13_grad/Reshape_1)^gradients_1/add_13_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_13_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_13_grad/MatMulMatMul0gradients_1/add_13_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_13_grad/MatMul_1MatMul	concat_120gradients_1/add_13_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
}
+gradients_1/MatMul_13_grad/tuple/group_depsNoOp"^gradients_1/MatMul_13_grad/MatMul$^gradients_1/MatMul_13_grad/MatMul_1
�
3gradients_1/MatMul_13_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_13_grad/MatMul,^gradients_1/MatMul_13_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_13_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_13_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_13_grad/MatMul_1,^gradients_1/MatMul_13_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_13_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_12_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_12_grad/modFloorModconcat_12/axisgradients_1/concat_12_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_12_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_12_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_12_grad/ConcatOffsetConcatOffsetgradients_1/concat_12_grad/mod gradients_1/concat_12_grad/Shape"gradients_1/concat_12_grad/Shape_1* 
_output_shapes
::*
N
�
 gradients_1/concat_12_grad/SliceSlice3gradients_1/MatMul_13_grad/tuple/control_dependency'gradients_1/concat_12_grad/ConcatOffset gradients_1/concat_12_grad/Shape*
_output_shapes

:
*
Index0*
T0
�
"gradients_1/concat_12_grad/Slice_1Slice3gradients_1/MatMul_13_grad/tuple/control_dependency)gradients_1/concat_12_grad/ConcatOffset:1"gradients_1/concat_12_grad/Shape_1*
Index0*
T0*
_output_shapes

:


{
+gradients_1/concat_12_grad/tuple/group_depsNoOp!^gradients_1/concat_12_grad/Slice#^gradients_1/concat_12_grad/Slice_1
�
3gradients_1/concat_12_grad/tuple/control_dependencyIdentity gradients_1/concat_12_grad/Slice,^gradients_1/concat_12_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/concat_12_grad/Slice*
_output_shapes

:

�
5gradients_1/concat_12_grad/tuple/control_dependency_1Identity"gradients_1/concat_12_grad/Slice_1,^gradients_1/concat_12_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/concat_12_grad/Slice_1*
_output_shapes

:


�
!gradients_1/Tanh_11_grad/TanhGradTanhGradTanh_115gradients_1/concat_12_grad/tuple/control_dependency_1*
_output_shapes

:

*
T0
n
gradients_1/add_12_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
   
   
p
gradients_1/add_12_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_12_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_12_grad/Shapegradients_1/add_12_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_12_grad/SumSum!gradients_1/Tanh_11_grad/TanhGrad-gradients_1/add_12_grad/BroadcastGradientArgs*
T0*
_output_shapes

:

*
	keep_dims( *

Tidx0
�
gradients_1/add_12_grad/ReshapeReshapegradients_1/add_12_grad/Sumgradients_1/add_12_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_12_grad/Sum_1Sum!gradients_1/Tanh_11_grad/TanhGrad/gradients_1/add_12_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
!gradients_1/add_12_grad/Reshape_1Reshapegradients_1/add_12_grad/Sum_1gradients_1/add_12_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_12_grad/tuple/group_depsNoOp ^gradients_1/add_12_grad/Reshape"^gradients_1/add_12_grad/Reshape_1
�
0gradients_1/add_12_grad/tuple/control_dependencyIdentitygradients_1/add_12_grad/Reshape)^gradients_1/add_12_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_12_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_12_grad/tuple/control_dependency_1Identity!gradients_1/add_12_grad/Reshape_1)^gradients_1/add_12_grad/tuple/group_deps*
_output_shapes

:
*
T0*4
_class*
(&loc:@gradients_1/add_12_grad/Reshape_1
�
!gradients_1/MatMul_12_grad/MatMulMatMul0gradients_1/add_12_grad/tuple/control_dependencyVariable_4/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
#gradients_1/MatMul_12_grad/MatMul_1MatMul	concat_110gradients_1/add_12_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
}
+gradients_1/MatMul_12_grad/tuple/group_depsNoOp"^gradients_1/MatMul_12_grad/MatMul$^gradients_1/MatMul_12_grad/MatMul_1
�
3gradients_1/MatMul_12_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_12_grad/MatMul,^gradients_1/MatMul_12_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_12_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_12_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_12_grad/MatMul_1,^gradients_1/MatMul_12_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_12_grad/MatMul_1*
_output_shapes

:

a
gradients_1/concat_11_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_1/concat_11_grad/modFloorModconcat_11/axisgradients_1/concat_11_grad/Rank*
T0*
_output_shapes
: 
q
 gradients_1/concat_11_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
s
"gradients_1/concat_11_grad/Shape_1Const*
valueB"
   
   *
dtype0*
_output_shapes
:
�
'gradients_1/concat_11_grad/ConcatOffsetConcatOffsetgradients_1/concat_11_grad/mod gradients_1/concat_11_grad/Shape"gradients_1/concat_11_grad/Shape_1* 
_output_shapes
::*
N
�
 gradients_1/concat_11_grad/SliceSlice3gradients_1/MatMul_12_grad/tuple/control_dependency'gradients_1/concat_11_grad/ConcatOffset gradients_1/concat_11_grad/Shape*
_output_shapes

:
*
Index0*
T0
�
"gradients_1/concat_11_grad/Slice_1Slice3gradients_1/MatMul_12_grad/tuple/control_dependency)gradients_1/concat_11_grad/ConcatOffset:1"gradients_1/concat_11_grad/Shape_1*
_output_shapes

:

*
Index0*
T0
{
+gradients_1/concat_11_grad/tuple/group_depsNoOp!^gradients_1/concat_11_grad/Slice#^gradients_1/concat_11_grad/Slice_1
�
3gradients_1/concat_11_grad/tuple/control_dependencyIdentity gradients_1/concat_11_grad/Slice,^gradients_1/concat_11_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/concat_11_grad/Slice*
_output_shapes

:

�
5gradients_1/concat_11_grad/tuple/control_dependency_1Identity"gradients_1/concat_11_grad/Slice_1,^gradients_1/concat_11_grad/tuple/group_deps*
_output_shapes

:

*
T0*5
_class+
)'loc:@gradients_1/concat_11_grad/Slice_1
�
!gradients_1/Tanh_10_grad/TanhGradTanhGradTanh_105gradients_1/concat_11_grad/tuple/control_dependency_1*
T0*
_output_shapes

:


n
gradients_1/add_11_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
p
gradients_1/add_11_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
-gradients_1/add_11_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_11_grad/Shapegradients_1/add_11_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_11_grad/SumSum!gradients_1/Tanh_10_grad/TanhGrad-gradients_1/add_11_grad/BroadcastGradientArgs*
_output_shapes

:

*
	keep_dims( *

Tidx0*
T0
�
gradients_1/add_11_grad/ReshapeReshapegradients_1/add_11_grad/Sumgradients_1/add_11_grad/Shape*
T0*
Tshape0*
_output_shapes

:


�
gradients_1/add_11_grad/Sum_1Sum!gradients_1/Tanh_10_grad/TanhGrad/gradients_1/add_11_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
!gradients_1/add_11_grad/Reshape_1Reshapegradients_1/add_11_grad/Sum_1gradients_1/add_11_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_1/add_11_grad/tuple/group_depsNoOp ^gradients_1/add_11_grad/Reshape"^gradients_1/add_11_grad/Reshape_1
�
0gradients_1/add_11_grad/tuple/control_dependencyIdentitygradients_1/add_11_grad/Reshape)^gradients_1/add_11_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_11_grad/Reshape*
_output_shapes

:


�
2gradients_1/add_11_grad/tuple/control_dependency_1Identity!gradients_1/add_11_grad/Reshape_1)^gradients_1/add_11_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_11_grad/Reshape_1*
_output_shapes

:

�
!gradients_1/MatMul_11_grad/MatMulMatMul0gradients_1/add_11_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
T0*
_output_shapes

:
*
transpose_a( 
�
#gradients_1/MatMul_11_grad/MatMul_1MatMul	concat_100gradients_1/add_11_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
}
+gradients_1/MatMul_11_grad/tuple/group_depsNoOp"^gradients_1/MatMul_11_grad/MatMul$^gradients_1/MatMul_11_grad/MatMul_1
�
3gradients_1/MatMul_11_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_11_grad/MatMul,^gradients_1/MatMul_11_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_11_grad/MatMul*
_output_shapes

:

�
5gradients_1/MatMul_11_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_11_grad/MatMul_1,^gradients_1/MatMul_11_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_11_grad/MatMul_1*
_output_shapes

:

�
gradients_1/AddNAddN2gradients_1/add_20_grad/tuple/control_dependency_12gradients_1/add_19_grad/tuple/control_dependency_12gradients_1/add_18_grad/tuple/control_dependency_12gradients_1/add_17_grad/tuple/control_dependency_12gradients_1/add_16_grad/tuple/control_dependency_12gradients_1/add_15_grad/tuple/control_dependency_12gradients_1/add_14_grad/tuple/control_dependency_12gradients_1/add_13_grad/tuple/control_dependency_12gradients_1/add_12_grad/tuple/control_dependency_12gradients_1/add_11_grad/tuple/control_dependency_1*
T0*4
_class*
(&loc:@gradients_1/add_20_grad/Reshape_1*
N
*
_output_shapes

:

�
gradients_1/AddN_1AddN5gradients_1/MatMul_20_grad/tuple/control_dependency_15gradients_1/MatMul_19_grad/tuple/control_dependency_15gradients_1/MatMul_18_grad/tuple/control_dependency_15gradients_1/MatMul_17_grad/tuple/control_dependency_15gradients_1/MatMul_16_grad/tuple/control_dependency_15gradients_1/MatMul_15_grad/tuple/control_dependency_15gradients_1/MatMul_14_grad/tuple/control_dependency_15gradients_1/MatMul_13_grad/tuple/control_dependency_15gradients_1/MatMul_12_grad/tuple/control_dependency_15gradients_1/MatMul_11_grad/tuple/control_dependency_1*
_output_shapes

:
*
T0*6
_class,
*(loc:@gradients_1/MatMul_20_grad/MatMul_1*
N


beta1_power_1/initial_valueConst*
_class
loc:@Variable_4*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable_4*
	container 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: *
use_locking(
m
beta1_power_1/readIdentitybeta1_power_1*
T0*
_class
loc:@Variable_4*
_output_shapes
: 

beta2_power_1/initial_valueConst*
_output_shapes
: *
_class
loc:@Variable_4*
valueB
 *w�?*
dtype0
�
beta2_power_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable_4
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: 
m
beta2_power_1/readIdentitybeta2_power_1*
T0*
_class
loc:@Variable_4*
_output_shapes
: 
�
!Variable_4/Adam/Initializer/zerosConst*
_class
loc:@Variable_4*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_4/Adam
VariableV2*
shared_name *
_class
loc:@Variable_4*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_4
y
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4*
_output_shapes

:

�
#Variable_4/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_4*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_4/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_4*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:

}
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_output_shapes

:
*
T0*
_class
loc:@Variable_4
�
!Variable_5/Adam/Initializer/zerosConst*
_class
loc:@Variable_5*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_5/Adam
VariableV2*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_5
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes

:

y
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes

:
*
T0*
_class
loc:@Variable_5
�
#Variable_5/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_5*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_5/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_5*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
}
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes

:

�
!Variable_6/Adam/Initializer/zerosConst*
_class
loc:@Variable_6*
valueB
*    *
dtype0*
_output_shapes

:

�
Variable_6/Adam
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_6*
	container *
shape
:

�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:
*
use_locking(
y
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes

:

�
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*
_class
loc:@Variable_6*
valueB
*    
�
Variable_6/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_6*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:
*
use_locking(
}
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes

:

�
!Variable_7/Adam/Initializer/zerosConst*
_class
loc:@Variable_7*
valueB*    *
dtype0*
_output_shapes
:
�
Variable_7/Adam
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes
:
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes
:*
T0*
_class
loc:@Variable_7
�
#Variable_7/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_7*
valueB*    *
dtype0*
_output_shapes
:
�
Variable_7/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@Variable_7
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes
:*
T0*
_class
loc:@Variable_7
Y
Adam_1/learning_rateConst*
_output_shapes
: *
valueB
 *��8*
dtype0
Q
Adam_1/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
Q
Adam_1/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
"Adam_1/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_1*
T0*
_class
loc:@Variable_4*
use_nesterov( *
_output_shapes

:
*
use_locking( 
�
"Adam_1/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( *
_output_shapes

:

�
"Adam_1/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon5gradients_1/MatMul_21_grad/tuple/control_dependency_1*
_output_shapes

:
*
use_locking( *
T0*
_class
loc:@Variable_6*
use_nesterov( 
�
"Adam_1/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon2gradients_1/add_21_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*
_class
loc:@Variable_7*
use_nesterov( 
�

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1#^Adam_1/update_Variable_4/ApplyAdam#^Adam_1/update_Variable_5/ApplyAdam#^Adam_1/update_Variable_6/ApplyAdam#^Adam_1/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable_4*
_output_shapes
: 
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: 
�
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2#^Adam_1/update_Variable_4/ApplyAdam#^Adam_1/update_Variable_5/ApplyAdam#^Adam_1/update_Variable_6/ApplyAdam#^Adam_1/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable_4*
_output_shapes
: 
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: 
�
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1#^Adam_1/update_Variable_4/ApplyAdam#^Adam_1/update_Variable_5/ApplyAdam#^Adam_1/update_Variable_6/ApplyAdam#^Adam_1/update_Variable_7/ApplyAdam
�
init_1NoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_6/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_7/Assign^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign""�
trainable_variables��
J

Variable:0Variable/AssignVariable/read:02Variable/initial_value:08
R
Variable_1:0Variable_1/AssignVariable_1/read:02Variable_1/initial_value:08
R
Variable_2:0Variable_2/AssignVariable_2/read:02Variable_2/initial_value:08
R
Variable_3:0Variable_3/AssignVariable_3/read:02Variable_3/initial_value:08
R
Variable_4:0Variable_4/AssignVariable_4/read:02Variable_4/initial_value:08
R
Variable_5:0Variable_5/AssignVariable_5/read:02Variable_5/initial_value:08
R
Variable_6:0Variable_6/AssignVariable_6/read:02Variable_6/initial_value:08
R
Variable_7:0Variable_7/AssignVariable_7/read:02Variable_7/initial_value:08"
train_op

Adam
Adam_1"�
	variables��
J

Variable:0Variable/AssignVariable/read:02Variable/initial_value:08
R
Variable_1:0Variable_1/AssignVariable_1/read:02Variable_1/initial_value:08
R
Variable_2:0Variable_2/AssignVariable_2/read:02Variable_2/initial_value:08
R
Variable_3:0Variable_3/AssignVariable_3/read:02Variable_3/initial_value:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
`
Variable/Adam:0Variable/Adam/AssignVariable/Adam/read:02!Variable/Adam/Initializer/zeros:0
h
Variable/Adam_1:0Variable/Adam_1/AssignVariable/Adam_1/read:02#Variable/Adam_1/Initializer/zeros:0
h
Variable_1/Adam:0Variable_1/Adam/AssignVariable_1/Adam/read:02#Variable_1/Adam/Initializer/zeros:0
p
Variable_1/Adam_1:0Variable_1/Adam_1/AssignVariable_1/Adam_1/read:02%Variable_1/Adam_1/Initializer/zeros:0
h
Variable_2/Adam:0Variable_2/Adam/AssignVariable_2/Adam/read:02#Variable_2/Adam/Initializer/zeros:0
p
Variable_2/Adam_1:0Variable_2/Adam_1/AssignVariable_2/Adam_1/read:02%Variable_2/Adam_1/Initializer/zeros:0
h
Variable_3/Adam:0Variable_3/Adam/AssignVariable_3/Adam/read:02#Variable_3/Adam/Initializer/zeros:0
p
Variable_3/Adam_1:0Variable_3/Adam_1/AssignVariable_3/Adam_1/read:02%Variable_3/Adam_1/Initializer/zeros:0
R
Variable_4:0Variable_4/AssignVariable_4/read:02Variable_4/initial_value:08
R
Variable_5:0Variable_5/AssignVariable_5/read:02Variable_5/initial_value:08
R
Variable_6:0Variable_6/AssignVariable_6/read:02Variable_6/initial_value:08
R
Variable_7:0Variable_7/AssignVariable_7/read:02Variable_7/initial_value:08
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
h
Variable_4/Adam:0Variable_4/Adam/AssignVariable_4/Adam/read:02#Variable_4/Adam/Initializer/zeros:0
p
Variable_4/Adam_1:0Variable_4/Adam_1/AssignVariable_4/Adam_1/read:02%Variable_4/Adam_1/Initializer/zeros:0
h
Variable_5/Adam:0Variable_5/Adam/AssignVariable_5/Adam/read:02#Variable_5/Adam/Initializer/zeros:0
p
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0
h
Variable_6/Adam:0Variable_6/Adam/AssignVariable_6/Adam/read:02#Variable_6/Adam/Initializer/zeros:0
p
Variable_6/Adam_1:0Variable_6/Adam_1/AssignVariable_6/Adam_1/read:02%Variable_6/Adam_1/Initializer/zeros:0
h
Variable_7/Adam:0Variable_7/Adam/AssignVariable_7/Adam/read:02#Variable_7/Adam/Initializer/zeros:0
p
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0�֮k