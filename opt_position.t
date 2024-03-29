local M = Dim("M",0)

local w_fitSqrt         = Param("w_fitSqrt", float, 0)
local w_regSqrt         = Param("w_regSqrt", float, 1)
local w_confSqrt        = 0.1
local Offset            = Unknown("Offset", opt_float3,{M},2)				    
local Angle             = Array("Angle", opt_float3,{M},3)		    		
local RobustWeights     = Array("RobustWeights", opt_float,{M},4)	
local UrShape           = Array("UrShape", opt_float3, {M},5)		   
local Constraints       = Array("Constraints", opt_float3,{M},6)	    
local ConstraintNormals = Array("ConstraintNormals", opt_float3,{M},7)
local TriWeights	    = Array("TriWeights", opt_float3, {M}, 8)
local G                 = Graph("G", 9, "w", {M}, 10, "v0", {M}, 11, "v1", {M}, 12, "v2", {M}, 13, 
                                "v3", {M}, 14, "v4", {M}, 15, "v5", {M}, 16, "v6", {M}, 17, "v7", {M}, 18, "c", {M}, 19, "n", {M}, 20, "r", {M}, 21)
local RegGrid           = Graph("RegGrid", 22, "v0", {M}, 23, "v1", {M}, 24)

UsePreconditioner(true)

--trilinear interpolation
local px00 = (1.0 - TriWeights(G.w)[0])*Offset(G.v0) + TriWeights(G.w)[0]*Offset(G.v4)
local px01 = (1.0 - TriWeights(G.w)[0])*Offset(G.v1) + TriWeights(G.w)[0]*Offset(G.v5)
local px10 = (1.0 - TriWeights(G.w)[0])*Offset(G.v2) + TriWeights(G.w)[0]*Offset(G.v6)
local px11 = (1.0 - TriWeights(G.w)[0])*Offset(G.v3) + TriWeights(G.w)[0]*Offset(G.v7)

local pxx0 = (1.0 - TriWeights(G.w)[1])*px00 + TriWeights(G.w)[1]*px10
local pxx1 = (1.0 - TriWeights(G.w)[1])*px01 + TriWeights(G.w)[1]*px11

local px = (1.0 - TriWeights(G.w)[2])*pxx0 + TriWeights(G.w)[2]*pxx1

local e_fit = RobustWeights(G.r) * ConstraintNormals(G.n):dot(px - Constraints(G.c))

local validConstraint = greatereq(Constraints(G.c), -9999.9)
Energy(w_fitSqrt*Select(validConstraint, e_fit, 0.0))

--RobustWeight Penalty
--local e_conf = 1-(RobustWeights(G.r)*RobustWeights(G.r))
--e_conf = Select(validConstraint, e_conf, 0.0)
--Energy(w_confSqrt*e_conf)

--regularization
local ARAPCost = (Offset(RegGrid.v0) - Offset(RegGrid.v1)) - Rotate3D(Angle(RegGrid.v0),UrShape(RegGrid.v0) - UrShape(RegGrid.v1))
Energy(w_regSqrt*ARAPCost)

