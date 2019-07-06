import math
from fractions import Fraction
import numbers


# Set of operation O as defined in the article
def inputs_are_numbers(a, b):
  return isinstance(a, numbers.Number) and isinstance(b, numbers.Number)


def Id(s):
  return s


def Add(a, b):
    if inputs_are_numbers(a, b):
        return a + b 
    return None

  
def Subtract(a, b):
    if inputs_are_numbers(a, b):
        return a - b
    return None

  
def Multiply(a, b):
    if inputs_are_numbers(a, b):
        return a * b
    return None
  
  
def Divide(a, b):
    if inputs_are_numbers(a, b) and b != 0:
        return a / b
    return None
  
    
def Power(a, b):
    if inputs_are_numbers(a, b):
        try:
            return a ** b
        except: pass
    return None

  
def Log(a):
    try:
        return math.log(a)
    except:
        return None

      
def Sqrt(a):
    try:
        return math.sqrt(a)
    except:
        return None
    
    
def Sine(a):
    try:
        return round(math.sin(Deg_to_Rad(a)), 3)
    except:
        return None
    
    
def Cosine(a):
    try:
        return round(math.cos(Deg_to_Rad(a)), 3)
    except:
        return None
    
    
def Tangent(a):
    try:
        return round(math.tan(Deg_to_Rad(a)), 3)
    except:
        return None

      
def Rad_to_Deg(a):
    try:
        return math.degrees(a)
    except:
        return None
    
    
def Deg_to_Rad(a):
    try:
        return math.radians(a)
    except:
        return None
    
    
def Factorial(a):
    try:
        res = 1
        while a > 1:
          res *= a
          a -= 1
        return res
    except:
        return None
    
    
def Choose(a, b):
    try:
        return (Factorial(a)) / (Factorial(b) * Factorial(a-b))
    except:
        return None


def Str_to_Float(s):
  return eval(s)


def Float_to_Str(a):
  return str(a)


def Str_to_Frac(s):
  return Str_to_Float(s.split('/')[0]) / Str_to_Float(s.split('/')[1])


def Frac_to_Str(a):
  return str(Fraction(a))


def LongNumber_to_Str(a):
  str_a = str(a)
  res = ''
  index = len(str_a)-1
  pos = 0
  while index >= 0:
    if pos > 0 and pos % 3 == 0:
      res += ','
    res += str_a[index]
    index -= 1
    pos += 1
  return res[::-1]


def Str_to_LongNumber(s):
  return float(s.replace('.', '').replace(',', ''))


def Check(s, options):
  check_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
  contains = []
  for option in options:
    if s in option:
      contains.append(check_dict[options.index(option)])
  
  if len(contains) != 1:
    return None
  else:
    return contains[0]

  
def apply(f, *args):
  if len(args) == argc[f]:
    return f(*args)
  return None
  
  
# O and argc for the operations in O
O = [Id, Add, Subtract, Multiply, Divide, 
     Power, Log, Sqrt, Sine, Cosine, Tangent, 
     Factorial, Choose, Rad_to_Deg, Deg_to_Rad, 
     Str_to_Float, Float_to_Str, Str_to_Frac, Frac_to_Str,
     LongNumber_to_Str, Str_to_LongNumber, Check]
argc = {Id: 1, Add: 2, Subtract: 2, Multiply: 2, Divide: 2, 
        Power: 2, Log: 1, Sqrt: 1, Sine: 1, Cosine: 1, Tangent: 1,
       Factorial: 1, Choose: 2, Rad_to_Deg: 1, Deg_to_Rad: 1, 
       Str_to_Float: 1, Float_to_Str: 1, Str_to_Frac: 1, Frac_to_Str: 1,
       LongNumber_to_Str: 1, Str_to_LongNumber: 1, Check: 2}
  
  
# Some tests
assert(apply(Add, 1, 2) == 3)
assert(apply(Subtract, 1, 2) == -1)
assert(apply(Multiply, 1, 2) == 2)
assert(apply(Divide, 1, 2) == 1/2)
assert(apply(Power, 1, 2) == 1)
assert(apply(Id, 'blat') == 'blat')
assert(argc[Log] == 1)
assert(argc[Add] == 2)
assert(Str_to_LongNumber('1,000,000') == 1000000)
assert(Str_to_LongNumber('1,000') == 1000)
assert(Str_to_LongNumber('100') == 100)
assert(Factorial(0) == 1)
assert(Factorial(1) == 1)
assert(Factorial(5) == 120)
assert(Str_to_Frac('3/4') == 3/4)
assert(Frac_to_Str(3/4) == '3/4')
assert(LongNumber_to_Str(1000000) == '1,000,000')
assert(LongNumber_to_Str(1000) == '1,000')
assert(LongNumber_to_Str(10000) == '10,000')
assert(LongNumber_to_Str(100) == '100')
assert(Cosine(180) == -1)
assert(Cosine(30) == 0.866)
assert(Cosine(90) == 0)
assert(Cosine(0) == 1)
assert(Tangent(180) == 0)
assert(Tangent(45) == 1)
assert(Tangent(30) == 0.577)
assert(Sine(180) == 0)
assert(Sine(90) == 1)
assert(Sine(45) == 0.707)
assert(Sine(30) == 0.5)
assert(Log(math.e**6) == 6)
assert(Choose(3, 2) == 3)
assert(Choose(52, 2) == 1326)
assert(Check('13', ["A ) 7", "B ) 9", "C ) 13", "D ) 27", "E ) 45"]) == 'C')
assert(Check('45', ["A ) 7", "B ) 9", "C ) 13", "D ) 27", "E ) 45"]) == 'E')
assert(Check('0', ["A ) 7", "B ) 9", "C ) 13", "D ) 27", "E ) 45"]) == None)
assert(Check('7', ["A ) 7", "B ) 9", "C ) 7", "D ) 27", "E ) 45"]) == None)

print("Yay!")
