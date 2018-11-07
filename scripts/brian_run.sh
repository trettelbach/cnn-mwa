#!/bin/bash

docker run --rm -it -v ~/cnn-mwa/data_test:/media/tabea/FIRENZE/cnn-mwa/data -v ~/cnn-mwa/data_test/images:/media/tabea/FIRENZE/cnn-mwa/data_test/images -v `pwd`:/workspace -w /workspace bskjerven/pytorch-test bash
