# DessiLBI
To initialize the toolbox, the following codes are needed.
```python
from slbi_toolbox import SLBI_ToolBox
import torch
optimizer = SLBI_ToolBox(model.parameters(), lr=args.lr, kappa=args.kappa, mu=args.mu, weight_decay=0)
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)
optimizer.print_network()
```  

For training a neural network, the process is similar to one that uses built-in optimizer
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```  

For pruning a neural network, the code is as follows.   

```python
optimizer.update_prune_order(epoch)
optimizer.prune_layer_by_order_by_list(percent, layer_name)
```   
