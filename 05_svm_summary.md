# Support Vector Machine(SVM)


아래의 그림에서 ① 분류기는 Train set을 틀리게 분류한다. 이를 여러번 학습시켜 모델링하면 ②와 ③ 분류기와 같이 될것이다. (여기서 분류기는 Hyperplane을 의미한다.)

Train set 측면에서 보면 ②와 ③ 분류기는 오류가 0이므로 같은 성능을 가진 분류기로 볼 수 있다. 하지만, 일반화(generalization) 측면에서 보면 ② 보다 ③이 더 낫다고 할 수 있다. 그 이유는 ③ 분류기가 두 개의 클래스에 대해 **여백**(margin) 크기 때문이다.

바로 여기서 이러한 여백, 즉 마진을 어떻게 공식화하고 이 마진을 최대화하는 결정 초평면(decision hyperplane)을 찾는 것이 바로 SVM의 발상이라 할 수 있다.



Margin을 최대화 한다
 = 일반화 오류 최소화
 = 좋은 성능



<img src="http://i.imgur.com/DrcoGVQ.png" width="350">

#### Margin
- Margin: 각 클래스에서 가장 가까운 관측치 사이의 거리
- Margin은 w(기울기)로 표현가능

## 2. 목적함수
다음과 같이 Margin은 각 기울기로 표현가능 하다. 두 클래스의 Margin을 최대화 하는 $\mathrm{w}^T + b+ =0$인 hyperplane을 찾는 것이 목적.

<img src="images/svm/margin.png" width="350">



$\mathrm{w}^Tx^ + + b  = 1$ ($x^+$는 plus-plane 위의 점)

$\mathrm{w}^T(x^- + \lambda) + b = 1$  ( $x^+ = x^ + \lambda \mathrm{w}$)

$\mathrm{w}^Tx^- + b + \lambda \mathrm{w}^T\mathrm{w} = 1$

$-1 + \lambda \mathrm{w}^T w 1$ ($x^-$는 \minus-plane 위의 점)

$\lambda = \frac{2}{\mathrm{w}^Tw}$ ($w$는 찾고자 하는 hyperplane의 기울기)


### Norm(노름)

$$
\lVert W \rVert_p = (\sum_{i}{| x |^p})^{1/p}
$$

#### $L_2$ norm
$$
\lVert W \rVert_2 = (\sum_{i}{| w_i |^2})^{1/2} = \sqrt{w_1^2+w_2^2+ ... w_n^2} = \sqrt{\mathrm{w}^t\mathrm w}
$$

</br>

그러므로, 다음과 같은 Margin 식을 얻을 수 있다.
$$
Margin = distance(x^+, x^-) \\
= \lVert{x^+ - x^-}\rVert_2\\
= \lVert{( x^+ + \lambda w)+ - x^-}\rVert_2\\
= ||\lambda w||_2\\
= \lambda \sqrt{\mathrm{w}^tw}\\
= \frac{2}{\mathrm{w}^tw} * \sqrt{\mathrm{w}^tw}\\
= \frac{2}{\sqrt{\mathrm{w}^tw}} = \frac{2}{\rVert{w}\rVert_2}
$$

결론으로, $\frac{2}{\rVert{w}\rVert_2}$ 을 Margin을 최대화 하는 $\mathrm w$ 값을 찾고 싶은 것이다. 이는 다음 식으로 나타 낼 수 있다.

$$
\max. \  margin = \max\frac{2}{\rVert{w}\rVert_2} \Leftrightarrow \min\frac{1}{2}\rVert{w}\rVert_2
$$

w의 $l_2$ norm은 제곱근을 포함, 계산상의 편의를 위해 제곱 형태의 목적 함수를 변경 한다.

$$
\min\frac{1}{2}\rVert{w}\rVert_2 \Leftrightarrow \min\frac{1}{2}\rVert{w}\rVert_2^2
$$

위의 문제의 두 가지 측면에서 보면 다음과 같다.

- **해의 유일성(uniqueness)**
    - $\mathbf{w}$의 2차항만 가지므로 볼록(convex)한 함수이며, 조건식은 $n$개 모두 선형이다. 따라서, 유일한 해를 가진다.

- **문제의 난이도**
    - 위의 문제는 $n$개의 선형 부등식을 가진 2차함수의 최적화 문제이며, $n$의 개수는 엄청나게 클 수 있다. 이러한 경우에 **라그랑제 승수**(Lagrange multiplier)를 이용해 해결할 수 있다.

## Convex Optimization Problem

$$
\min \frac{1}{2}\lVert \mathrm{w} \rVert_2^2 \\
\text{s.t.} \  y_i (\mathrm{w}^Tx_i + b)≥1, i = 1,2,\ ...\ ..., n
$$

- Decision Variable은 W와 b
- Objective Function은 Separating hyperplane으로 부터 정의된 margin의 역수
- Constraint는 training data를 **완벽하게** seperating하는 조건(<u>margin이 1보다 큰 경우</u>)
- Objective function은 quadratic이고, constraint 선형이다. $\rightarrow$ Quadratic Programming(QP)$\rightarrow$ Convex Optimization $\rightarrow$ 전역 최적해가 존재 한다.
- Training Data가 linearly separable한 경우에만 존재


## Lagrangian Formulation
라그랑주 승수를 이용하여 Lagrangian primal 문제로 변환 한다. 먼저, 간단한 예로 라그랑제 승수법을 살펴보도록 하자.



아래와 같이 **등식 제약**의 조건 하에서 함수 $f(x,y)$를 최소화 하는 $x,y$를 찾는 문제를 살표보면 다음과 같다.

라그랑주 승수법은 두 그래프의 각 방향의 그라디언트가 같아지는 지점이 최대 지점이다. 기울기는 같다고 크기는 다를 수 있으니 $\lambda$를 곱해주는 것이다. 이를 라그랑주 승수라고 한다. 공식은 다음과 같다.
$$
g(a,b) = 0\\
\nabla f(a,b) = \lambda\nabla g(a,b)
$$

위 식을 라그랑제 승수법을 사용하여 **라그랑제 함수**로 다음과 같이 정의가 가능하다.
$$
\min f(x,y) = x^2 + 2y \\
\text{s.t.} \ \  3x + 2y + 1 = 0
$$

$$
L(x,y,\alpha) = f(x,y) - \lambda(3x + 2y + 1 ) = x^2 + 2y - \lambda(3x + 2y + 1)
$$

$(\hat{x},\hat{y})$가 제약이 있는 최적화 문제의 해라면 $(\hat{x},\hat{y}, \hat{\lambda})$가  함수 $L(\cdot)$ 의 정류점(stationary point) 즉, 모든 편도함수(partial derivative)가 0인 지점이 되는 $\lambda$가 존재 한다는 것을 보였다.

따라서,$x,y,\lambda$ ​에 대한 라그랑지안 ​의 편도함수를 계산할 수 있으면 이 도함수가 모두 0이 되는 지점을 찾을 수 있고, 제약이 있는 최적화 문제의 해는 이런 정류점 중에 있어야 한다. 위의 식의 편도함수는 다음과 같다.
$$
\frac{\partial}{\partial x} L(x,y,\lambda) = 2x - 3\lambda \\
$$

$$
\frac{\partial}{\partial y} L(x,y,\lambda) = 2 - 2\lambda \\
$$

$$
\frac{\partial}{\partial \lambda} L(x,y,\lambda) = -3x -2y -1
$$

위 식을 게산 하면, $\hat{x} = \frac{3}{2}, \hat{y} = \frac{11}{4}, \hat{\lambda} = 1$을 얻을 수 있다.

### 라그랑제 함수로 변환

위의 조건식에서 각 조건식 마다 라그랑제 승수 $\alpha_i$를 부여한다. 이들의 벡터를 $\mathbf{\alpha} = (\alpha_1, \dots , \alpha_n)^{T}$로 표기하자.

$$
L(\mathbf{w}, b, \mathbf{\alpha}) = \frac{1}{2} \| \mathbf{w} \|^{2} - \sum_{i=1}^{n}{\alpha_{i} \left( t_{i} \left( \mathbf{w}^{T} \mathbf{x}_{i} +b \right) -1 \right)}
$$

### Original Problem
$$
\min\frac{1}{2}\rVert{w}\rVert_2^2 \newline
subject\ to\ y_i(\mathrm{w}^TX_i + b) ≥ 1,\ i = 1,2,\ ...\ ...\ n
$$

### Lagrangian Primal
$$
\underset{a}{\max}\  \underset{\mathrm{w},b}{\min} \ L(\mathrm{w},b,a) = \min\frac{1}{2}\rVert{w}\rVert_2^2 - \sum_{i=1}^{n}a_i(y_i(\mathrm{w}^{T}x_i + b) -1 ) \\
subject\ to\ a_i ≥ 0,\ i = 1,2,\ ...\ ... \ ,n
$$
Convex, Continuous(연속형)이기 때문에 미분 = 0에서 최소 값을 갖는다.

$\frac{\partial L(\mathrm{w},b,a)}{\partial \mathrm{w}} = 0 \rightarrow w = \sum_{i=0}^{n}a_iy_ix_i$
2. $\frac{\partial L(\mathrm{w},b,a)}{\partial \mathrm{b}} = 0 \rightarrow \sum_{i=0}^{n}a_iy_i = 0$

### Lagrangian Formulation

#### 1. $\min\frac{1}{2}\rVert{w}\rVert_2^2$
$$
\min\frac{1}{2}\rVert{\mathrm{w}}\rVert_2^2 = \frac{1}{2}\mathrm{w^Tw}\\
= \frac{1}{2}\mathrm{w}^T \sum_{j=1}^n a_jy_jx_j\\
= \frac{1}{2}\sum_{j=1}^n a_j{y_j}(\mathrm w^Tx_j)\\
= \frac{1}{2}\sum_{j=1}^n a_j{y_j}(\sum_{i=1}^{n} a_jy_jx_i^Tx_j)\\
= \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n a_ia_jy_iy_jx_i^Tx_j
$$

### Lagrangian Dual
다음과 같이 식을 정리 가능하다.
$$
\min\frac{1}{2}\rVert{w}\rVert_2^2 - \sum_{i=1}^{n}a_i(y_i(\mathrm{w}^{T}x_i + b) -1 )\\
= \sum_{i=1}^{n} a_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}a_ia_jy_iy_jx_i^Tx_j\ (\text{where} \sum_{i=1}^{n}a_iy_i = 0)
$$


위의 식은 다음과 같은 특징이 있다.

- 하나의 등식 조건과 $n$개의 부등식 조건을 가진 **2차(quadratic)** 목적함수의 최대화 문제이다.
- $\mathbf{w}$와 $b$가 사라졌으므로, 라그랑제 승수 $\mathbf{\alpha}$를 구하는 문제가 되었다. 따라서, $\mathbf{\alpha}$를 구하면 $\mathbf{w}$와 $b$를 구할 수 있다.
- 목적함수에서 특성 벡터 $\mathbf{x}_i$가 **내적** $\mathbf{x}_{i}^{T} \cdot \mathbf{x}_{j}$로 나타난다. 이러한 점은 선형 SVM을 비선형 SVM으로 즉, **Kernel SVM**으로 확장하는데 결정적인 역할을 한다.

## 2) KKT(Karush-Kuhn-Tucker) 조건

위에서 살펴본 예제는 등식 제약 조건부 최적화 문제였다. 하지만, SVM은 **부등식 제약**(inequality constrained) 최적화 문제이다(예를들어 $3x + 2y + 1 \ge 0$).

이러한 부등식 제약 최적화 문제는 다음과 같이 세 가지 **KKT**조건을 이용해 풀수 있다. KKT 조건은 **필요 조건** 이므로 반드시 만족해야 한다.

- 라그랑제 함수 $L(\mathbf{w}, b, \mathbf{\alpha})$에서 라그라제 승수를 제외한 $\mathbf{w}, b$로 편미분한  식이 $0$이되어야 한다.

$$
\frac{\partial L(\mathbf{w}, b, \mathbf{\alpha})}{\partial \mathbf{w}} = \frac{\partial L(\mathbf{w}, b, \mathbf{\alpha})}{\partial b} = 0
$$

- 모든 라그랑제 승수 $\mathbf{\alpha} = \{ \alpha_1, \dots, \alpha_n \}$은 0보다 크거나 같아야 한다.

$$
\alpha_i \ge 0, \quad i = 1, \dots, n
$$

- 모든 조건식에 대해 $\alpha_i = 0$ 이거나 $t_{i} \left( \mathbf{w}^{T} \mathbf{x}_{i} +b \right) -1 = 0$ 이 되어야 한다. 이때, $t_{i} \left( \mathbf{w}^{T} \mathbf{x}_{i} +b \right) = 1$인 데이터 포인트가 바로 **서포트 벡터**이다.





## Reference
- [서포트 벡터 머신 (Support Vector Machine) · ratsgo's blog](https://ratsgo.github.io/machine%20learning/2017/05/23/SVM/)

- [(3) [핵심 머신러닝] SVM 모델 1 (Margin, Hard Margin Linear SVM) - YouTube](https://www.youtube.com/watch?v=qFg8cDnqYCI)




