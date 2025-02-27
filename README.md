# Template match with gray model(ncc)

## rotate-model vs main branch


| method | main(ms) | rotate-model(ms) | factor(main/rotate-model) |
|--------|----------|------------------|--------|
|train|1|680|1/680|     
|match|31|16|2|
|train-omp|1|160|1/160|
|match-omp|12|6|2|

result: main brain mathod train model really fast(1ms), rotate-model method 2x faster in matching.