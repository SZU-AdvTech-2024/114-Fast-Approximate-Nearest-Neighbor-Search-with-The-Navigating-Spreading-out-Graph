数据集下载链接：
https://xlliu-beihang.github.io/hashing/dataset.html

./test_nndescent sift_base.fvecs sift.200NN.graph 200 200 10 10 100

cd nsg/build/tests

./test_nsg_index efanna_graph/tests/sift_base.fvecs efanna_graph/tests/sift.200NN.graph 40 50 500 sift.nsg
 
./test_nsg_optimized_search efanna_graph/tests/sift_base.fvecs efanna_graph/tests/sift_query.fvecs sift.nsg 70 50 nsg/search_result.ivecs
./test_nsg_search efanna_graph/tests/sift_base.fvecs efanna_graph/tests/sift_query.fvecs sift.nsg 70 50 nsg/search_result.ivecs