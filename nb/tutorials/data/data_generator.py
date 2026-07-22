import numpy as np
import torch
import sympy as sp

custom_seed = 42
np.random.seed(custom_seed)
torch.manual_seed(custom_seed)

class ExampleScalarFieldTwoDimensions:

    def __init__(self, N_data:int, f_char:str, dx:float, dy:float, indexing):
        """
        Initializes the ExampleScalarFieldTwoDimensions object. 
        This class is used to create a dataset that is a scalar field in two dimensions. It is understood as a zero-order tensor.
        In addition to creating the field, its gradient and its laplacian will also be created.

        Args:
        - N_data (int): The number of data samples.
        - f (str): A string that contains the analytical function of the field.
        - dx (float): Grid spacing in the x direction.
        - dy (float): Grid spacing in the y direction.
        """
        self.N_data = N_data
        self.dx = dx
        self.dy = dy

        self.tensorial_order = [0, 0] # [Contravariant, covariant]
        self.dim = 2 # f in R2
        self.f_char = f_char

        self.xr = np.arange(-2, 2,self.dx)
        self.yr = np.arange(-2, 2,self.dy)
        self.xx, self.yy = np.meshgrid(self.xr, self.yr, sparse=True, indexing=indexing)

        self.sym_functions()

    def mesh(self):
        """
        Returns the meshgrid coordinates.

        Returns:
        X (ndarray): X coordinate meshgrid.
        Y (ndarray): Y coordinate meshgrid.
        """
        X = self.xx
        Y = self.yy

        return X, Y
    
    def sym_functions(self):
        """
        Defines the scalar field function and its derivatives. 
        
        The function f(x, y) is defined symbolically. 
        Then its derivatives are made and transformed so that they are all evaluated numerically. 
        
        In this way, to obtain later the gradient and the Laplacian, you only have to change f(x, y) (it has to be a simple function supported by sympy)
        because they are derived automatically, you do not have to declare from zero each of the derivatives.
        """
        x, y = sp.symbols('x y')
        
        # Define the function
        f_sym = sp.sympify(self.f_char)

        # Compute the partial derivatives
        dfdx_sym = sp.diff(f_sym, x)
        dfdy_sym = sp.diff(f_sym, y)

        # Compute the Laplacian
        laplacian_sym = sp.diff(f_sym, x, x) + sp.diff(f_sym, y, y)

        # Convert symbolic expressions to numerical functions
        self.f_num = sp.lambdify((x, y), f_sym, 'numpy')
        self.dfdx_num = sp.lambdify((x, y), dfdx_sym, 'numpy')
        self.dfdy_num = sp.lambdify((x, y), dfdy_sym, 'numpy')
        self.laplacian_num = sp.lambdify((x, y), laplacian_sym, 'numpy')

    def func(self):
        """
        Computes the scalar field function values on the grid.

        Returns:
        torch.Tensor: A tensor containing the scalar field values with shape (N_data, 1, len(xr), len(yr)).
        """
        X = self.xx
        Y = self.yy
        f = self.f_num

        f_field = np.zeros((self.N_data, 1, len(self.xr), len(self.yr)), dtype=np.float32)
        f_field[:, 0, :, :] = f(X, Y)

        return torch.tensor(f_field, dtype=torch.float32)

    def gradient(self):
        """
        Computes the gradient of the scalar field function on the grid.

        Returns:
        torch.Tensor: A tensor containing the gradient components with shape (N_data, 3, len(xr), len(yr)).
        """
        X = self.xx
        Y = self.yy
        dfdx, dfdy = self.dfdx_num, self.dfdy_num

        grad_field = np.zeros((self.N_data, 2, len(self.xr), len(self.yr)), dtype=np.float32)
        grad_field[:, 0, :, :] = dfdx(X, Y)
        grad_field[:, 1, :, :] = dfdy(X, Y)

        return torch.tensor(grad_field, dtype=torch.float32)

    def laplacian(self):
        """
        Computes the Laplacian of the scalar field function on the grid.

        Returns:
        torch.Tensor: A tensor containing the Laplacian values with shape (N_data, 1, len(xr), len(yr)).
        """
        X = self.xx
        Y = self.yy
        lap = self.laplacian_num

        lap_field = np.zeros((self.N_data, 1, len(self.xr), len(self.yr)), dtype=np.float32)
        lap_field[:, 0, :, :] = lap(X, Y)

        return torch.tensor(lap_field, dtype=torch.float32)


class ExampleScalarFieldThreeDimensions:
    
    def __init__(self, N_data:int, f_char:str, dx:float, dy:float, dz:float, indexing):   
            """
            Initializes the ExampleScalarFieldThreeDimensions object. 
            This class is used to create a dataset that is a scalar field in three dimensions. It is understood as a zero-order tensor.
            In addition to creating the field, its gradient and its laplacian will also be created.

            Args:
            - N_data (int): The number of data samples.
            - f_char (str): A string that contains the analytical function of the field.
            - dx (float): Grid spacing in the x direction.
            - dy (float): Grid spacing in the y direction.
            - dz (float): Grid spacing in the z direction.
            """
            
            self.N_data = N_data
            self.dx = dx
            self.dy = dy
            self.dz = dz

            self.tensorial_order = [0, 0] # [Contravariant, covariant]
            self.dim = 3 # f in R3
            self.f_char = f_char

            self.xr = np.arange(-2, 2,self.dx)
            self.yr = np.arange(-2, 2,self.dy)
            self.zr = np.arange(-2, 2,self.dz)
            self.xx, self.yy, self.zz = np.meshgrid(self.xr, self.yr, self.zr, sparse=True, indexing=indexing)

            self.sym_functions()

    def mesh(self):
        """
        Returns the meshgrid coordinates.

        Returns:
        X (ndarray): X coordinate meshgrid.
        Y (ndarray): Y coordinate meshgrid.
        """
        X = self.xx
        Y = self.yy
        Z = self.zz

        return X, Y, Z
    
    def sym_functions(self):
        """
        Defines the scalar field function and its derivatives. 
        
        The function f(x, y) is defined symbolically. 
        Then its derivatives are made and transformed so that they are all evaluated numerically. 
        
        In this way, to obtain later the gradient and the Laplacian, you only have to change f(x, y) (it has to be a simple function supported by sympy)
        because they are derived automatically, you do not have to declare from zero each of the derivatives.
        """
        x, y, z = sp.symbols('x y z')

        # Define the function
        f_sym = sp.sympify(self.f_char)
    
        # Compute the partial derivatives
        dfdx_sym = sp.diff(f_sym, x)
        dfdy_sym = sp.diff(f_sym, y)
        dfdz_sym = sp.diff(f_sym, z)

        # Compute the Laplacian
        laplacian_sym = sp.diff(f_sym, x, x) + sp.diff(f_sym, y, y) + sp.diff(f_sym, z, z)

        # Convert symbolic expressions to numerical functions
        self.f_num = sp.lambdify((x, y, z), f_sym, 'numpy')
        self.dfdx_num = sp.lambdify((x, y, z), dfdx_sym, 'numpy')
        self.dfdy_num = sp.lambdify((x, y, z), dfdy_sym, 'numpy')
        self.dfdz_num = sp.lambdify((x, y, z), dfdz_sym, 'numpy')
        self.laplacian_num = sp.lambdify((x, y, z), laplacian_sym, 'numpy')

    def func(self):
        """
        Computes the scalar field function values on the grid.

        Returns:
        torch.Tensor: A tensor containing the scalar field values with shape (N_data, 1, len(xr), len(yr)).
        """
        X = self.xx
        Y = self.yy
        Z = self.zz
        f = self.f_num

        f_field = np.zeros((self.N_data, 1, len(self.xr), len(self.yr), len(self.zr)), dtype=np.float32)
        f_field[:, 0, :, :] = f(X, Y, Z)

        return torch.tensor(f_field, dtype=torch.float32)

    def gradient(self):
        """
        Computes the gradient of the scalar field function on the grid.

        Returns:
        torch.Tensor: A tensor containing the gradient components with shape (N_data, 3, len(xr), len(yr)).
        """
        X = self.xx
        Y = self.yy
        Z = self.zz
        dfdx, dfdy, dfdz = self.dfdx_num, self.dfdy_num, self.dfdz_num

        grad_field = np.zeros((self.N_data, 3, len(self.xr), len(self.yr), len(self.zr)), dtype=np.float32)
        grad_field[:, 0, :, :, :] = dfdx(X, Y, Z)
        grad_field[:, 1, :, :, :] = dfdy(X, Y, Z)
        grad_field[:, 2, :, :, :] = dfdz(X, Y, Z)

        return torch.tensor(grad_field, dtype=torch.float32)

    def laplacian(self):
        """
        Computes the Laplacian of the scalar field function on the grid.

        Returns:
        torch.Tensor: A tensor containing the Laplacian values with shape (N_data, 1, len(xr), len(yr)).
        """
        X = self.xx
        Y = self.yy
        Z = self.zz
        lap = self.laplacian_num

        lap_field = np.zeros((self.N_data, 1, len(self.xr), len(self.yr), len(self.zr)), dtype=np.float32)
        lap_field[:, 0, :, :] = lap(X, Y, Z)

        return torch.tensor(lap_field, dtype=torch.float32)


class ExampleVectorialFieldTwoDimensions:

    def __init__(self, N_data:int, Fx_char:str, Fy_char:str, dx:float, dy:float, indexing):   
        """
        Initializes the ExampleVectorialFieldTwoDimensions object. 
        This class is used to create a dataset that is a vectorial field in two dimensions. It is understood as a first-order tensor.
        In addition to creating the field, its gradient and its laplacian will also be created.

        Args:
        - N_data (int): The number of data samples.
        - Fx_char (str): A string that contains the analytical function of the first component of the vectorial field.
        - Fy_char (str): A string that contains the analytical function of the second component of the vectorial field.
        - dx (float): Grid spacing in the x direction.
        - dy (float): Grid spacing in the y direction.
        """
        self.N_data = N_data
        self.dx = dx
        self.dy = dy

        self.tensorial_order = [1, 0] # [Contravariant, covariant]
        self.dim = 2 # f in R2
        self.Fx_char = Fx_char
        self.Fy_char = Fy_char

        self.xr = np.arange(-2, 2,self.dx)
        self.yr = np.arange(-2, 2,self.dy)
        self.xx, self.yy = np.meshgrid(self.xr, self.yr, sparse=True, indexing=indexing)

        self.sym_functions()

    def mesh(self):
        """
        Returns the meshgrid coordinates.

        Returns:
        X (ndarray): X coordinate meshgrid.
        Y (ndarray): Y coordinate meshgrid.
        """
        X = self.xx
        Y = self.yy

        return X, Y

    def sym_functions(self):
        """
        Defines the vectorial field function and its derivatives. 
        
        The functions Fi(x, y) (i = x,y) are defined symbolically. 
        Then its derivatives are made and transformed so that they are all evaluated numerically. 
        
        In this way, to obtain later the gradient and the Laplacian, you only have to change Fi(x, y) (i = x,y) (it has to be a simple function supported by sympy)
        because they are derived automatically, you do not have to declare from zero each of the derivatives.
        """
        x, y = sp.symbols('x y')
        
        # Define the components of the vectorial field and convert to numerical functions
        Fx_sym = sp.sympify(self.Fx_char)
        Fy_sym = sp.sympify(self.Fy_char)

        self.Fx_num = sp.lambdify((x, y), Fx_sym, 'numpy')
        self.Fy_num = sp.lambdify((x, y), Fy_sym, 'numpy')

        # Compute the partial derivatives and convert to numerical functions
        self.dFxdx_num = sp.lambdify((x, y), sp.diff(Fx_sym, x), 'numpy')
        self.dFxdy_num = sp.lambdify((x, y), sp.diff(Fx_sym, y), 'numpy')

        self.dFydx_num = sp.lambdify((x, y), sp.diff(Fy_sym, x), 'numpy')
        self.dFydy_num = sp.lambdify((x, y), sp.diff(Fy_sym, y), 'numpy')

        # Compute the Laplacian and convert to numerical functions
        self.Lapx_num = sp.lambdify((x, y), sp.diff(Fx_sym, x, x) + sp.diff(Fx_sym, y, y), 'numpy')
        self.Lapy_num = sp.lambdify((x, y), sp.diff(Fy_sym, x, x) + sp.diff(Fy_sym, y, y), 'numpy')

    def Func(self):
        """
        Compute the vector field at the mesh grid points.
        
        Returns:
        torch.Tensor: A tensor containing the vectorial field values with shape (N_data, 3, len(xr), len(yr), len(zr)).
        """
        X, Y = self.xx, self.yy
        Fx = self.Fx_num(X, Y)
        Fy = self.Fy_num(X, Y)

        F_field = np.zeros((self.N_data, 2, len(self.xr), len(self.yr)), dtype=np.float32)
        F_field[:, 0, :, :] = Fx
        F_field[:, 1, :, :] = Fy

        return torch.tensor(F_field, dtype=torch.float32)

    def gradient(self):
        """
        Compute the gradient of the vector field.
        
        Returns:
        torch.Tensor: A tensor containing the gradient components with shape (N_data, 3, 3, len(xr), len(yr), len(zr)).
        """
        X, Y = self.xx, self.yy

        grad_field = np.zeros((self.N_data, 2, 2, len(self.xr), len(self.yr)), dtype=np.float32)
        grad_field[:, 0, 0, :, :] = self.dFxdx_num(X, Y)
        grad_field[:, 0, 1, :, :] = self.dFxdy_num(X, Y)
        
        grad_field[:, 1, 0, :, :] = self.dFydx_num(X, Y)
        grad_field[:, 1, 1, :, :] = self.dFydy_num(X, Y)

        return torch.tensor(grad_field, dtype=torch.float32)

    def laplacian(self):
        """
        Compute the Laplacian of the vector field.
        
        Returns:
        torch.Tensor: A tensor containing the Laplacian values with shape (N_data, 3, len(xr), len(yr)).
        """
        X, Y = self.xx, self.yy

        laplacian_field = np.zeros((self.N_data, 2, len(self.xr), len(self.yr)), dtype=np.float32)
        laplacian_field[:, 0, :, :] = self.Lapx_num(X, Y)
        laplacian_field[:, 1, :, :] = self.Lapy_num(X, Y)

        return torch.tensor(laplacian_field, dtype=torch.float32)


class ExampleVectorialFieldThreeDimensions:

    def __init__(self, N_data:int, Fx_char:str, Fy_char:str, Fz_char:str, dx:float, dy:float, dz:float, indexing):   
        """
        Initializes the ExampleVectorialFieldThreeDimensions object. 
        This class is used to create a dataset that is a vectorial field in two dimensions. It is understood as a first-order tensor.
        In addition to creating the field, its gradient and its laplacian will also be created.

        Args:
        - N_data (int): The number of data samples.
        - Fx_char (str): A string that contains the analytical function of the first component of the vectorial field.
        - Fy_char (str): A string that contains the analytical function of the second component of the vectorial field.
        - Fz_char (str): A string that contains the analytical function of the third component of the vectorial field.
        - dx (float): Grid spacing in the x direction.
        - dy (float): Grid spacing in the y direction.
        - dz (float): Grid spacing in the z direction.
        """
        self.N_data = N_data
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.tensorial_order = [1, 0] # [Contravariant, covariant]
        self.dim = 3 # f in R2
        self.Fx_char = Fx_char
        self.Fy_char = Fy_char
        self.Fz_char = Fz_char

        self.xr = np.arange(-2, 2, self.dx)
        self.yr = np.arange(-2, 2, self.dy)
        self.zr = np.arange(-2, 2, self.dz)
        self.xx, self.yy, self.zz = np.meshgrid(self.xr, self.yr, self.zr, sparse=True, indexing=indexing)

        self.sym_functions()

    def mesh(self):
        """
        Returns the meshgrid coordinates.

        Returns:
        X (ndarray): X coordinate meshgrid.
        Y (ndarray): Y coordinate meshgrid.
        Z (ndarray): Z coordinate meshgrid.
        """
        X = self.xx
        Y = self.yy
        Z = self.zz

        return X, Y, Z

    def sym_functions(self):
        """
        Defines the scalar field function and its derivatives. 
        
        The functions F(x, y, z) are defined symbolically. 
        Then its derivatives are made and transformed so that they are all evaluated numerically. 
        
        In this way, to obtain later the gradient and the Laplacian, you only have to change F(x, y, z) (it has to be a simple function supported by sympy) 
        because they are derived automatically, you do not have to declare from zero each of the derivatives.
        """
        x, y, z = sp.symbols('x y z')
        
        # Define the components of the vectorial field and convert to numerical functions
        Fx_sym = sp.sympify(self.Fx_char)
        Fy_sym = sp.sympify(self.Fy_char)
        Fz_sym = sp.sympify(self.Fz_char)

        # Compute the partial derivatives
        dFxdx_sym = sp.diff(Fx_sym, x)
        dFxdy_sym = sp.diff(Fx_sym, y)
        dFxdz_sym = sp.diff(Fx_sym, z)

        dFydx_sym = sp.diff(Fy_sym, x)
        dFydy_sym = sp.diff(Fy_sym, y)
        dFydz_sym = sp.diff(Fy_sym, z)

        dFzdx_sym = sp.diff(Fz_sym, x)
        dFzdy_sym = sp.diff(Fz_sym, y)
        dFzdz_sym = sp.diff(Fz_sym, z)

        # Compute the Laplacian
        Lapx_sym = sp.diff(Fx_sym, x, x) + sp.diff(Fx_sym, y, y) + sp.diff(Fx_sym, z, z)
        Lapy_sym = sp.diff(Fy_sym, x, x) + sp.diff(Fy_sym, y, y) + sp.diff(Fy_sym, z, z)
        Lapz_sym = sp.diff(Fz_sym, x, x) + sp.diff(Fz_sym, y, y) + sp.diff(Fz_sym, z, z)

        # Convert symbolic expressions to numerical functions
        self.Fx_num = sp.lambdify((x, y, z), Fx_sym, 'numpy')
        self.Fy_num = sp.lambdify((x, y, z), Fy_sym, 'numpy')
        self.Fz_num = sp.lambdify((x, y, z), Fz_sym, 'numpy')

        self.dFxdx_num = sp.lambdify((x, y, z), dFxdx_sym, 'numpy')
        self.dFxdy_num = sp.lambdify((x, y, z), dFxdy_sym, 'numpy')
        self.dFxdz_num = sp.lambdify((x, y, z), dFxdz_sym, 'numpy')

        self.dFydx_num = sp.lambdify((x, y, z), dFydx_sym, 'numpy')
        self.dFydy_num = sp.lambdify((x, y, z), dFydy_sym, 'numpy')
        self.dFydz_num = sp.lambdify((x, y, z), dFydz_sym, 'numpy')

        self.dFzdx_num = sp.lambdify((x, y, z), dFzdx_sym, 'numpy')
        self.dFzdy_num = sp.lambdify((x, y, z), dFzdy_sym, 'numpy')
        self.dFzdz_num = sp.lambdify((x, y, z), dFzdz_sym, 'numpy')

        self.Lapx_num = sp.lambdify((x, y, z), Lapx_sym, 'numpy')
        self.Lapy_num = sp.lambdify((x, y, z), Lapy_sym, 'numpy')
        self.Lapz_num = sp.lambdify((x, y, z), Lapz_sym, 'numpy')
                
    def Func(self):
        """
        Computes the vectorial field function values on the grid.

        Returns:
        torch.Tensor: A tensor containing the vectorial field values with shape (N_data, 3, len(xr), len(yr), len(zr)).
        """
        X = self.xx
        Y = self.yy
        Z = self.zz

        Fx = self.Fx_num
        Fy = self.Fy_num
        Fz = self.Fz_num

        F_field = np.zeros((self.N_data, 3, len(self.xr), len(self.yr), len(self.zr)), dtype=np.float32)
        F_field[:, 0, :, :, :] = Fx(X, Y, Z)
        F_field[:, 1, :, :, :] = Fy(X, Y, Z)
        F_field[:, 2, :, :, :] = Fz(X, Y, Z)

        return torch.tensor(F_field, dtype=torch.float32)
    
    def gradient(self):
        """
        Computes the gradient of the vectorial field function on the grid.

        Returns:
        torch.Tensor: A tensor containing the gradient components with shape (N_data, 3, 3, len(xr), len(yr), len(zr)).
        """
        X = self.xx
        Y = self.yy
        Z = self.zz

        dFxdx = self.dFxdx_num
        dFxdy = self.dFxdy_num
        dFxdz = self.dFxdz_num

        dFydx = self.dFydx_num
        dFydy = self.dFydy_num
        dFydz = self.dFydz_num

        dFzdx = self.dFzdx_num
        dFzdy = self.dFzdy_num
        dFzdz = self.dFzdz_num

        grad_field = np.zeros((self.N_data, 3, 3, len(self.xr), len(self.yr), len(self.zr)), dtype=np.float32)
        grad_field[:, 0, 0, :, :, :] = dFxdx(X, Y, Z)
        grad_field[:, 0, 1, :, :, :] = dFxdy(X, Y, Z)
        grad_field[:, 0, 2, :, :, :] = dFxdz(X, Y, Z)
        
        grad_field[:, 1, 0, :, :, :] = dFydx(X, Y, Z)
        grad_field[:, 1, 1, :, :, :] = dFydy(X, Y, Z)
        grad_field[:, 1, 2, :, :, :] = dFydz(X, Y, Z)
        
        grad_field[:, 2, 0, :, :, :] = dFzdx(X, Y, Z)
        grad_field[:, 2, 1, :, :, :] = dFzdy(X, Y, Z)
        grad_field[:, 2, 2, :, :, :] = dFzdz(X, Y, Z)

        return torch.tensor(grad_field, dtype=torch.float32)
    
    def laplacian(self):
        """
        Computes the Laplacian of the vectorial field function on the grid.

        Returns:
        torch.Tensor: A tensor containing the Laplacian values with shape (N_data, 3, len(xr), len(yr)).
        """
        X = self.xx
        Y = self.yy
        Z = self.zz

        Lapx = self.Lapx_num
        Lapy = self.Lapy_num
        Lapz = self.Lapz_num

        laplacian_field = np.zeros((self.N_data, 3, len(self.xr), len(self.yr), len(self.zr)), dtype=np.float32)
        laplacian_field[:, 0, :, :, :] = Lapx(X, Y, Z)
        laplacian_field[:, 1, :, :, :] = Lapy(X, Y, Z)
        laplacian_field[:, 2, :, :, :] = Lapz(X, Y, Z)

        return torch.tensor(laplacian_field, dtype=torch.float32)