# NER_metrics
Some scripts for evaluating a NER model

一些ner模型指标评估的脚本，评价过程的具体区别可参考[Named-Entity evaluation metrics based on entity-level]( http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/ )。

- conlleval.pl  CONLL-2003评价脚本，只考虑实体类型与位置完全匹配的情况。
- conlleval.py 上面脚本的py版本，来自https://github.com/spyysalo/conlleval.py。
- metric.py  在 [NCRF++]( https://github.com/jiesutd/NCRFpp ) 评价方法的基础上，引入IOU指标来考虑预测结果与标注结果模糊匹配的情况。在某些实际应用过程中能够更客观的反映模型的性能指标。

