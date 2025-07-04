import matplotlib.pyplot as plt
import torch

gelu, relu = torch.nn.GELU() , torch.nn.ReLU()

x = torch.linspace(-3, 3 ,100)
y_gelu , y_relu = gelu(x) , relu(x)
plt.figure(figsize=(8,3))
for i , (y , label) in enumerate(zip([y_gelu, y_relu] , ["GELU", "ReLU"]),1):
    plt.subplot(1,2,i)
    plt.plot(x,y)
    plt.title( f"{label} activation furction")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()


