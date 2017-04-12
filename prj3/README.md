# MiniProject 3 Report
* Tuan Nguyen
* Deep Learning Spring 2017
* Advisor: Dr. Martin Hagan

## 1.Download code/dataset
Note: file `ptb_word_lm.py` is actually `pdb_word_lm.py` 
## 2.Run the program ptb_word_lm.py
Using iPython as following to run the code:

```
ipython
run pdb_word_lm.py --data_pathrun pdb_word_lm.py --data_path=simple-examples/data/ --model=small
```

Here is the output of the program:

```
0.004 perplexity: 54.999 speed: 12742 wps
0.104 perplexity: 40.762 speed: 13672 wps
0.204 perplexity: 44.566 speed: 13697 wps
0.304 perplexity: 42.803 speed: 13705 wps
0.404 perplexity: 42.056 speed: 13709 wps
0.504 perplexity: 41.435 speed: 13712 wps
0.604 perplexity: 40.064 speed: 13713 wps
0.703 perplexity: 39.446 speed: 13714 wps
0.803 perplexity: 38.806 speed: 13715 wps
0.903 perplexity: 37.490 speed: 13716 wps
Epoch: 13 Train Perplexity: 36.695
Epoch: 13 Valid Perplexity: 121.737
Test Perplexity: 116.733
```

## 3.Performance of the network
In this network, there are two criterias used to measure the performance of the network:
 * Perplexity
	```
	perplexity = np.exp(costs/iters)
	```
 * Word per second (wps)
	```
	wps = iters * batch_size / (current_time - start_time)
    ```
## 4.Network Size
In provided network, there are:
 * 2 layers/cells
 * 200 units/neurons per layers 
 * 2 * 200 = 400 units in total
### 4.1 Increase layers and units
With `3 layers`:
```
0.004 perplexity: 66.537 speed: 9316 wps
0.104 perplexity: 47.559 speed: 9890 wps
0.204 perplexity: 52.318 speed: 9906 wps
0.304 perplexity: 50.404 speed: 9911 wps
0.404 perplexity: 49.814 speed: 9913 wps
0.504 perplexity: 49.199 speed: 9915 wps
0.604 perplexity: 47.664 speed: 9916 wps
0.703 perplexity: 47.055 speed: 9916 wps 
0.803 perplexity: 46.383 speed: 9916 wps
0.903 perplexity: 44.895 speed: 9916 wps
Epoch: 13 Train Perplexity: 43.988     
Epoch: 13 Valid Perplexity: 123.126   
Test Perplexity: 118.102
```
With `2 layers and 400 units per layer`:

```
0.004 perplexity: 33.633 speed: 9466 wps
0.104 perplexity: 24.162 speed: 9995 wps
0.204 perplexity: 26.218 speed: 10005 wps
0.304 perplexity: 24.816 speed: 10010 wps
0.404 perplexity: 24.051 speed: 10014 wps
0.504 perplexity: 23.336 speed: 10015 wps
0.604 perplexity: 22.242 speed: 10016 wps
0.703 perplexity: 21.578 speed: 10017 wps
0.803 perplexity: 20.871 speed: 10018 wps
0.903 perplexity: 19.849 speed: 10019 wps
Epoch: 13 Train Perplexity: 19.180
Epoch: 13 Valid Perplexity: 152.214
Test Perplexity: 145.104
```
Comparing three networks:
 * 3 layers has higher perplexity (in all train, valid and test sets) than 2 layers, so increaseing the layers doesn't help
 * 2 layers and 400 units seems to be overfitting because the train perplexity is so good while the valid and test is much higher than others
 * 3 layers speed is the best while 2 layers and 200 units per layer is worst
