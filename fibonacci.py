#program to print n numbers in the fibonacci series
i=0
j=1
n=int(input("Enter the number of elements of Fibonacci series to be displayed- "))
if n<1:
      print("invalid, enter a positive no greater than 0")
k=0
while k<n:
     print(i)
     h=i+j
     i=j
     j=h
     k+=1 
     
    
