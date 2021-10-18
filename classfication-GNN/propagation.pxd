from eigency.core cimport *
from libcpp.string cimport string

ctypedef unsigned int uint

cdef extern from "agpAlg.cpp":
	pass

cdef extern from "agpAlg.h" namespace "propagation":
	cdef cppclass Agp:
		Agp() except+
		double agp_operation(string,string,uint,uint,int,double,double,double,Map[MatrixXd] &)