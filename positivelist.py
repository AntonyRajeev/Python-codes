#to find and print all positive numbers in a range
list=[]
n=int(input("enter the no of elements and the respective elements in the list  "))
for i in range(0,n):
      i=int(input())
      list.append(i)
print("The positive integers are -")
for k in list:
      if k>=0:
        print(k)
