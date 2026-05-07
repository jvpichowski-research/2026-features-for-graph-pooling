
# The Role of Node Features in Graph Pooling



## Overview

The script implements the experimental setup proposed in the paper, where graph pooling operators are evaluated under a unified framework that incorporates positional encodings (PEs) into the selection step. This allows comparison of pooling methods with and without PEs and various pooling methods.


## Installation

Install required dependencies:

```shell
pip install -r requirements.txt
```

## Usage

### Example Usage

```shell
python main.py -d MUTAG --pe none -p mincut
```

### Arguments

```
--seed, -s                  Seed: 1 (default)
--pooler, -p                Pooling method: none (default), mincut, diffpool, dmon, mdlpool, jbpool

--pe                        Positional encodings: none (default), laplacian, node2vec, rw, gpse

--pe-dim                    Dimension of positional encoding: 6 (default)

--hidden-channels, -c       Hidden layer dimension: 200 (default)

--dataset, -d               Dataset: PROTEINS (default), MUTAG, ENZYMES, COLLAB, IMDB-BINARY, 
                                     REDDIT-BINARY, COLORS-3, DD, Mutagenicity, NCI1

--output_dir                Output directory for results: results (default)

--draw_assignments          Visualize pooling assignments: False (default)
```



