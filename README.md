# Single Image FL

- **Fix for using the flwr simulation on GPU :**
    1. Find the ray_client_proxy.py in your local pkg installation directory of python packages under the folder "flwr/simulation/ray_transport/". 
    2. Replace @ray_remote() before 'launch_and_XXX' methods to @ray_remote(max_calls=1)
    3. Save the file and execute the main simulation python script of our project as usual from now on.