from functools import WRAPPER_ASSIGNMENTS, reduce
import math
#from __future__ import division
from collections import Counter
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pass
# zip 
# a= [1,2,3], b= ['a','b','c'] --> zip(a,b) = [(1,a),(2,b),(3,c)]
class vector:
    def __init__(self):
        pass
    def vector_add(v,w):
        return [v_i + w_i for v_i, w_i in zip(v,w)]
    
    def vector_subtract(v,w):
        return [v_i - w_i
            for v_i,w_i in zip(v,w)]
    
    def vector_sum(vectors):
        result = vectors[0]
        for vector in vectors[1:]:
            result = vector.vector_add(result,vector)
        #reduce(vector.vector_add,vectprs)
        # partial(reduce, vector_add)
        return result
    
    def scalar_multiply(c,v):
        """ c는 숫자 , v는 벡터 """
        return [c * v_i for v_i in v]
    
    def vector_mean(vectors):
        n = len(vectors)
        return vector.scalar_multiply(1/n, vector.vector_sum(vectors))

    # 벡터 내적
    def dot(v,w):
        """ v_1 * w_1 + ... + v_n * w_n"""
        return sum(v_i * w_i
        for v_i, w_i in zip(v,w))

    # 각 성분의 제곱의 합
    def sum_of_squares(v):
        """v_1 * v_1 + ... + v_n * v_n"""
        return vector.dot(v,v)

    # 벡터의 크기.
    def magnitude(v):
        return math.sqrt(vector.sum_of_squares(v))

    # 두 벡터간의 거리.
    def squared_distance(v,w):
        """(v_1 - w_1) ** 2 + .... (v_n - w_n) ** 2"""
        return vector.sum_of_squares(vector.vector_subtract(v,w))
    
    def distance(v,w):
        return math.sqrt(vector.squared_distance(v,w))
        # return manitude(vector.vector_subtract(v,w))

class matrix:
    def __init__(self):
        pass
    def shape(A):
        num_rows = len(A)
        num_cols = len(A[0]) # 첫번째 행이 가지고 있는 원소 수.
        return num_rows, num_cols

    def get_row(A, i):
        return A[i]
    
    def get_column(A,j):
        return [A_i[j]
            for A_i in A]
    
    ## list comprehension 에 대한 이해 필요.

    #entry_fn 행렬
    def make_matrix(num_rows, num_cols, entry_fn):
        """(i,j)번째 원소가 entry_fn(i,j)인
        num_rows x num_cols list를 반환"""
        return [[entry_fn(i,j)
            for j in range(num_cols)]
            for i in range(num_rows)]

    def is_diagonal(i,j) :
        """ 대각선의 원소는 1, 나머지 원소는 2 """
        return 1 if i == j else 2

    # identity_matrix  = make_matrix(5,5)


class statistics:
    def __init__(self):
        pass

    # 평균
    def mean(x):
        return sum(x) / len(x)

    def median(v):
        n = len(v)
        sorted_v = sorted(v)
        midpoint = n //2

        if n % 2 == 1:
            return sorted_v[midpoint]

    # p로 몇번째 정렬된 값을 호출
    def quantile(x,p):
        """x의 p 분위에 속하는 값을 반환"""
        p_index = int(p*len(x))
        return sorted(x) [p_index]

    # iteritems 옵션 확인 필요.
    def mode(x):
        """ 최빈값이 하나보다 많다면 list를 반환"""
        counts = Counter(x)
        max_count = max(counts.values())
        return [x_i for x_i, count in counts.iteritems()
            if count == max_count]

    def data_range(x):
        return max(x) - min(x)
    
    def de_mean(x):
        """ x의 모든 데이터 포인트에서 평균을 뺌(평균을 0으로 만들기 위해)"""
        x_bar = statistics.mean(x)
        return [x_i - x_bar for x_i in x]

    def variance(x):
        """ x에 두개 이상의 데이터 포인트가 존재한다고 가정"""
        n = len(x)
        deviations = statistics.de_mean(x)
        return vector.sum_of_squares(deviations)

    # 표준편차.
    def standard_deviation(x):
        return math.sqrt(statistics.variance(x))

    # 상위 25% - 하위 25%
    # 이상치 대비
    def interquartile_range(x):
        return statistics.quantile(x,0.75) - statistics.quantile(x,0.25)

    # 공분산
    #공분산은 X의 편차와 Y의 편차를 곱한것의 평균
    def covariance(x,y):
        n = len(x)
        return vector.dot(statistics.de_mean(x),statistics.de_mean(y)) / (n-1)
    
    # 상관관계 -> (-1 ~ 1)
    def correlation(x,y):
        stdev_x = statistics.standard_deviation(x)
        stdev_y = statistics.standard_deviation(y)
        if stdev_x > 0 and stdev_y > 0:
            return statistics.covariance(x,y) / stdev_x / stdev_y
        else:
            return 0


class probability:
    def __init__(self):
        pass

    #확률밀도함수.
    def uniform_pdf(x):
        return 1 if x>=0 and x< 1 else 0

    # 누적분포함수    
    def uniform_cdf(x):

        if x< 0: return 0
        elif x< 1: return x
        else : return 1

    # 정규분포.
    def normal_pdf(x, mu = 0, sigma=1):
        sqrt_two_pi = math.sqrt(2 * math.pi)
        return (math.exp(-(x-mu) ** 2 )) / (sqrt_two_pi * sigma)

    # 정규분포의 누적분포함수.
    # math.erf ( error function 사용하여)
    # 
    def normal_cdf(x, mu=0,sigma=1):
        return (1+ math.erf((x-mu) / math.sqrt(2) / sigma)) /2

    # 누적분포함수 의 역함수

    def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
        """이진 검색을 사용해서 역함수를 근사"""
        mid_z = 0
        # 표준정규분포가 아니라면 표준정규분포로 변환.

        if mu != 0 or sigma != 1:
            return mu + sigma * probability.inverse_normal_cdf(p, tolerance=tolerance)
        low_z, low_p = -10.0 , 0
        hi_z, hi_p = 10.0, 1
        while hi_z - low_z > tolerance:
            mid_z = (low_z + hi_z) /2
            mid_p = probability.normal_cdf(mid_z)
            if mid_p < p:
                low_z, low_p = mid_z, mid_p
            elif mid_p >p:
                hi_z, hi_p = mid_z, mid_p
            else:
                break
        return mid_z

    def bernoulli_trial(p):
        return 1 if random.random() < p else 0

    def binomial(n,p):
        return sum(probability.bernoulli_trial(p) for _ in range(n))

    def make_hist(p,n,num_points):
        data = [probability.binomial(n,p) for _ in range(num_points)]
        pass

class hypothesis:
    
    def __init__(self):
    # 누적 분포함수는 확률변수가 특정 값보다 작을 확률을 나타낸다.
        self.normal_probability_below = probability.normal_cdf

    def normal_approximation_to_binomial(n,p):
        """ Binomial(n,p)에 해당되는 mu(평균)와 sigma(표준편차) 계산"""
        mu = p *n
        # e(x^2) - e(x)^2 = 분산 // 베르누이 p x=1 / 1-p x=0
        sigma = math.sqrt(p* (1-p) *n)
        return mu, sigma

    # 만약 확률변수가 특정 값보다 작지 않다면, 특정 값보다 크다는 것을 의미한다. 
    def normal_probability_above(lo, mu = 0, sigma=1):
        return 1 - probability.normal_cdf(lo,mu,sigma)
    
    def normal_probability_between(lo,hi, mu=0, sigma=1):
        return probability.normal_cdf(hi,mu,sigma) - probability.normal_cdf(lo,mu, sigma)
    
    def normal_probability_outside(lo,hi, mu=0, sigma=1):
        return 1 - hypothesis.normal_probability_between(lo,hi,mu,sigma)

    def normal_upper_bound(probability, mu =0, sigma=1):
        """P(Z <= z) = probability 인 z 값을 반환"""
        return probability.inverse_normal_cdf(probability, mu, sigma)
    
    def normal_lower_bound(probability, mu =0, sigma=1):
        """P(Z >=z) = probability인 z 값을 반환"""
        return probability.inverse_normal_cdf(1 - probability,mu, sigma)
    
    def normal_two_sided_bounds(probability, mu =0 , sigma=1):
        """입력한 probability 값을 포함하고,
        평균을 중심으로 대칭적인 구간을 반환"""
        tail_probability = (1- probability) / 2
        
        # 구간의 상한은 tail_probability 값 이상의 확률 값을 갖고 있다.
        upper_bound = hypothesis.normal_lower_bound(tail_probability, mu, sigma)

        #구간의 하한은 tail_probability 값 이하의 확률 값을 갖고 있다.
        lower_bound = hypothesis.normal_upper_bound(tail_probability, mu, sigma)
    



class gradient:
    def __init__(self):
        pass

    def sum_of_squares(v):
        """v에 속해 있는 항목들의 제곱합을 계산한다."""
        return sum(v_i ** 2 for v_i in v)

    def difference_quotient(f,x,h):
        # f(x) = ax +b
        # f'(x)  =a 예시
        return (f(x+h) - f(x)) / h
    
    def square(x):
        return x*x
    
    # 도함수
    def derivative(x):
        return 2*x
    
    def partial_difference_quotient(f,v,i,h):
        """함수 f의 i번째 편도함수가 v에서 가지는 값"""
        w = [v_j + (h if j == i else 0)
            for j, v_j in enumerate(v)]
        
        return (f(w) - f(v)) /h

    def estimate_gradient(f,v,h=0.00001):
        return [gradient.partial_difference_quotient(f,v,i,h)
            for i, _ in enumerate(v)]

    def step(v, direction, step_size):
        """v에서 step_size만큼 이동하기"""
        return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v,direction)]

    def sum_of_squares_gradient(v):
        return [2 * v_i for v_i in v]

    """
    v = [random.randint(-10,10) for in range(3)]
    tolerance = 0.000001

    while True:
            gradient = sum_of_suqres_gradient(v)  #vd의 경사도 계산
            next_v = step(v, gradient, -0.01)  #경사도의 음수만큼 이동
            if distance(next_v,v) < tolerance:
                break
            v = next_v
    """

    # 적절한 이동 거리 정하기
    
    def safe(f):
        """ f와 또같은 함수를 반환하지만 F에 오류가 발생하면
        무한대를 반환해준다."""
        def safe_f(*args,**kwargs):
            try:
                return f(*args, **kwargs)
            except:
                return float('inf')
        return safe_f

    def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
        """ 목적 함수를 최소화시키는 theta를 경사 하강법을 사용해서 찾아본다."""
        step_sizes = [100,10,1,0.1,0.01,0.001,0.0001,0.00001]

        theta =theta_0 #theta를 시작점으로 설정
        target_fn = gradient.safe(target_fn) # 오류를 처리할 수 있는 target_fn으로 변환
        value = target_fn(theta) #최소화시키려는 값.

        while True:
            gradient = gradient_fn(theta)
            next_thetas = [gradient.step(theta, gradient, -step_size)
                for step_size in step_sizes]
        
            #함수를 최소화시키는 theta 선택
            next_theta = min(next_thetas, key= target_fn)
            next_value = target_fn(next_theta)

            # tolerance만큼 수렴하면 멈춤
            if abs(value - next_value) < tolerance:
                return theta
            else:
                theta, value = next_theta, next_value

    
    def negate(f):
        """ x를 입력하면 -f(x)를 반환해 주는 함수 생성"""
        return lambda *args, **kwargs: -f(*args, **kwargs)

    def negate_all(f):
        """f가 여러 숫자를 반환할 때 모든 숫자를 음수를 변환"""
        return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

    def maximize_batch(target_fn,gradient_fn, theta_0, tolerance=0.000001):
        return gradient.minimize_batch(gradient.negate(target_fn),
            gradient.negate_all(gradient_fn),
            theta_0,
            tolerance)
    #SGd (stochastic gradient descent)
    # 생략

class use_data:
    def __init__(self):
        pass

    def bucketize(point, bucket_size):
        """ 각 데이터를 bucket_size의 배수에 해당하는 구간에 위치시킨다."""
        return bucket_size * math.floor(point / bucket_size) # math.floor 반내림.

    def make_histogram(points, bucket_size):
        """ 구간을 생성하고 각 구간 내 데이터 개수를 계산해 준다."""
        return Counter(use_data.bucketize(point,bucket_size) for point in points)
    
    def plot_histogram(points,bucket_size, title=""):
        histogram = use_data.make_histogram(points, bucket_size)
        plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
        plt.title(title)
        plt.show()

    def random_normal():
        """표준정규분포를 따르는 임의의 데이터를 반환"""
        return probability.inverse_normal_cdf(random.random())

    def correlation_matrix(data):
        """(i,j)번째 항목이 i번째 차원과 j번째 차원의 상관관걔를 나타내는
         num_columns X num_columns 행렬 반환"""
        
        _, num_columns = matrix.shape(data)

        def matrix_entry(i,j):
            return statistics.correlation(matrix.get_column(data, i), matrix.get_column(data,j))

        return matrix.make_matrix(num_columns, num_columns, matrix_entry)

    def parse_row(input_row,parsers):
        """파서 list(None이 포함될 수도 있다.)가 주어지면
        각 input_row의 항목에 적절한 파서를 적용"""

        return [parser(value) if parser is not None else value
            for value,parser in zip(input_row,parsers)]
        
    def parse_rows_with(reader,parsers):
        """각 열에 파서를 적용하기 위해 reader를 치환"""
        for row in reader:
            yield gradient.parse_row(row,parsers)
