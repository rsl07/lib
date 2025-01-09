#!/home/d51680/Logiciel/anaconda3/envs/py_lionel/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:04:14 2022

@author: roussela
""" 

from sigfig                 import round
from math                   import floor, log10, atan, sqrt, pi
from anytree                import Node, RenderTree
from matplotlib.collections import PatchCollection

import os
import ipdb 

import pprint
import numpy  as np
import pandas as pd

import matplotlib.patches as patches
import matplotlib.pyplot  as plt


pp = pprint.PrettyPrinter().pprint


###############################################################################
                                #FORMAT#
###############################################################################

def lround(lst,num):
    if type(lst) != list :
        raise ValueError(str(type(lst))+" is not list")
        
    ls_rounded = [ round(l,num) for l in lst ]
    
    return ls_rounded




def around(arr,num):
    if type(arr) != np.ndarray :
        raise ValueError(str(type(arr))+" is not numpy.ndarrray")
    
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            
            arr[i,j] = round(arr[i,j],num)
    
    return arr



def float_formatter(sig_fig=4, len_str = 10, space=True, neg_space=False,resolution=10):
    """
    sig_fig    = number of significativ digit
    len_str    = length of the formatted string (coma excluded)
    resolution = minimal measurable value
    """

    res=resolution

    if type(sig_fig) not in [int, np.int]:
        raise ValueError("type(sig_fig) wrong :",type(sig_fig))

    if type(len_str) not in [int, np.int]:
        raise ValueError("type(len_str) wrong :",type(len_str))
        
    if type(space) != bool :
        raise ValueError("space must be a bool")
        
    if type(neg_space) != bool : 
        raise ValueError("space must be a bool")
    
    if space :
        sp = 1
    else :
        sp = 0
        
    if neg_space :
        ng_sp = 1
    else : 
        ng_sp = 0


    if res not in range(0,11):
        raise ValueError("resolution must be in range(0,11)")

        
        
    
    def ff(num):

        if num == 0 :
            l_sp = len_str - 1 - ng_sp

            return " "*ng_sp + '0' + l_sp*" "*sp


        num = round(num, sig_fig)      

        if -1000 < num and num <= -100 :

            pre  = sig_fig - 3

            # fmt  = "{:."+str(pre)+"f}"
            # l_sp = len_str - ng_sp - sig_fig -1

            # if "." not in fmt.format(num):
            #     l_sp += 1

            # return fmt.format(num) + l_sp*" "*sp

            if pre <= res :

                fmt  = "{:."+str(pre)+"f}"
                l_sp = len_str - ng_sp - sig_fig -1

                if "." not in fmt.format(num):
                    l_sp += 1

                return fmt.format(num) + l_sp*" "*sp

            else: 

                fmt  = "{:."+str(res)+"f}"
                l_sp = len_str - ng_sp - 3 - 1 - res

                if "." not in fmt.format(num):
                    l_sp += 1

                return fmt.format(num) + l_sp*" "*sp


        if -100 < num and num <= -10 :

            pre  = sig_fig - 2 

            # fmt  = "{:."+str(pre)+"f}"
            # l_sp = len_str - ng_sp - sig_fig -1
            
            # if "." not in fmt.format(num):
            #     l_sp += 1

            # return fmt.format(num) + l_sp*" "*sp

            if pre <= res :

                pre  = sig_fig - 2 
                fmt  = "{:."+str(pre)+"f}"
                l_sp = len_str - ng_sp - sig_fig -1
                
                if "." not in fmt.format(num):
                    l_sp += 1

                return fmt.format(num) + l_sp*" "*sp

            else: 

                fmt  = "{:."+str(res)+"f}"
                l_sp = len_str - ng_sp - 2 - 1 - res

                if "." not in fmt.format(num):
                    l_sp += 1

                return fmt.format(num) + l_sp*" "*sp



        elif -10 < num and num <= -1 :

            pre  = sig_fig - 1

            # fmt  = "{:."+str(pre)+"f}"
            # l_sp = len_str - ng_sp - sig_fig -1

            # if "." not in fmt.format(num):
            #     l_sp += 1

            # return fmt.format(num) + l_sp*" "*sp

            if pre <= res :

                fmt  = "{:."+str(pre)+"f}"
                l_sp = len_str - ng_sp - sig_fig -1
                
                if "." not in fmt.format(num):
                    l_sp += 1

                return fmt.format(num) + l_sp*" "*sp

            else: 

                fmt  = "{:."+str(res)+"f}"
                l_sp = len_str - ng_sp - 1 - 1 - res

                if "." not in fmt.format(num):
                    l_sp += 1

                return fmt.format(num) + l_sp*" "*sp
        


        elif -1 < num and num <= -0.1 :

            pre  = sig_fig 

            # fmt  = "{:."+str(pre)+"f}"
            # l_sp = len_str - ng_sp - sig_fig -1 - 1

            # return fmt.format(num) + l_sp*" "*sp


            if pre <= res :

                fmt  = "{:."+str(pre)+"f}"
                l_sp = len_str - ng_sp - sig_fig -1 - 1

                return fmt.format(num) + l_sp*" "*sp

            else: 

                fmt  = "{:."+str(res)+"f}"
                l_sp = len_str - ng_sp - 1 - 1 - res

                if "." not in fmt.format(num):
                    l_sp += 1

                return fmt.format(num) + l_sp*" "*sp


        elif -0.1 < num and num <= -0.01 :

            pre  = sig_fig+1

            # fmt  = "{:."+str(pre)+"f}"
            # l_sp = len_str - ng_sp - sig_fig -1 - 2 

            # return fmt.format(num) + l_sp*" "*sp

            if pre <= res :

                fmt  = "{:."+str(pre)+"f}"
                l_sp = len_str - ng_sp - sig_fig -1 - 2 

                return fmt.format(num) + l_sp*" "*sp

            else: 

                fmt  = "{:."+str(res)+"f}"
                l_sp = len_str - ng_sp - 1 - 1 - res

                if "." not in fmt.format(num):
                    l_sp += 1

                return fmt.format(num) + l_sp*" "*sp


        elif num == 0 or abs(num) <= 1e-15 :
            l_sp = len_str - 2 - ng_sp

            return " "*ng_sp + '0.' + l_sp*" "*sp


        elif 0.01 <= num and num < 0.1 :

            pre  = sig_fig+1

            # fmt  = "{:."+str(pre)+"f}"
            # l_sp = len_str - ng_sp - sig_fig -1 - 2

            # return " "*ng_sp + fmt.format(num) + l_sp*" "*sp


            if pre <= res :

                fmt  = "{:."+str(pre)+"f}"
                l_sp = len_str - ng_sp - sig_fig -1 - 2

                return " "*ng_sp + fmt.format(num) + l_sp*" "*sp

            else: 

                fmt  = "{:."+str(res)+"f}"
                l_sp = len_str - ng_sp - 1 - 1 - res

                if "." not in fmt.format(num):
                    l_sp += 1

                return " "*ng_sp +fmt.format(num) + l_sp*" "*sp


        elif 0.1 <= num and num < 1 :

            pre  = sig_fig 
            # fmt  = "{:."+str(pre)+"f}"
            # l_sp = len_str - ng_sp - sig_fig -1 - 1

            # return " "*ng_sp + fmt.format(num) + l_sp*" "*sp

            if pre <= res :

                fmt  = "{:."+str(pre)+"f}"
                l_sp = len_str - ng_sp - sig_fig -1 - 1

                return " "*ng_sp + fmt.format(num) + l_sp*" "*sp


            else: 

                fmt  = "{:."+str(res)+"f}"
                l_sp = len_str - ng_sp - 1 - 1 - res

                if "." not in fmt.format(num):
                    l_sp += 1

                return " "*ng_sp +fmt.format(num) + l_sp*" "*sp
        

        elif 1 <= num and num < 10 :

            pre  = sig_fig -1

            # fmt  = "{:."+str(pre)+"f}"
            # l_sp = len_str - ng_sp - sig_fig -1 

            # if "." not in fmt.format(num):
            #     l_sp += 1

            # return " "*ng_sp + fmt.format(num) + l_sp*" "*sp


            if pre <= res :

                fmt  = "{:."+str(pre)+"f}"
                l_sp = len_str - ng_sp - sig_fig -1 

                if "." not in fmt.format(num):
                    l_sp += 1

                return " "*ng_sp + fmt.format(num) + l_sp*" "*sp


            else: 

                fmt  = "{:."+str(res)+"f}"
                l_sp = len_str - ng_sp - 1 - 1 - res

                if "." not in fmt.format(num):
                    l_sp += 1

                return " "*ng_sp + fmt.format(num) + l_sp*" "*sp


        elif 10 <= num and num < 100 :

            pre  = sig_fig -2

            # fmt  = "{:."+str(pre)+"f}"
            # l_sp = len_str - ng_sp - sig_fig -1

            # if "." not in fmt.format(num):
            #     l_sp += 1

            # return " "*ng_sp + fmt.format(num) + l_sp*" "*sp


            if pre <= res :

                fmt  = "{:."+str(pre)+"f}"
                l_sp = len_str - ng_sp - sig_fig -1

                if "." not in fmt.format(num):
                    l_sp += 1

                return " "*ng_sp + fmt.format(num) + l_sp*" "*sp


            else: 

                fmt  = "{:."+str(res)+"f}"
                l_sp = len_str - ng_sp - 2 - 1 - res

                if "." not in fmt.format(num):
                    l_sp += 1

                return " "*ng_sp +fmt.format(num) + l_sp*" "*sp



        elif 100 <= num and num < 1000 :

            pre  = sig_fig - 3
            # fmt  = "{:."+str(pre)+"f}"
            # l_sp = len_str - ng_sp - sig_fig -1

            # if "." not in fmt.format(num):
            #     l_sp += 1
            
            # return " "*ng_sp + fmt.format(num) + l_sp*" "*sp


            if pre <= res :

                fmt  = "{:."+str(pre)+"f}"
                l_sp = len_str - ng_sp - sig_fig -1

                if "." not in fmt.format(num):
                    l_sp += 1
                
                return " "*ng_sp + fmt.format(num) + l_sp*" "*sp



            else: 

                fmt  = "{:."+str(res)+"f}"
                l_sp = len_str - ng_sp - 3 - 1 - res

                if "." not in fmt.format(num):
                    l_sp += 1

                return " "*ng_sp +fmt.format(num) + l_sp*" "*sp
                

        elif num < 0 :

            precision   = sig_fig - 1 


            if precision < res:

                e_formatter = "{:."+str(precision)+"e}"

                l_sp = len_str - 1 - (sig_fig+1) - 4 
                l_sp = max(l_sp, 0)

            else:

                e_formatter = "{:."+str(res)+"e}"

                l_sp = len_str - 1 - (2+res) - 4 
                l_sp = max(l_sp, 0)


            return e_formatter.format(num) + l_sp*" " *sp

        

        elif num > 0 :

            precision   = sig_fig - 1 

            # e_formatter = "{:."+str(precision)+"e}"

            # l_sp = len_str - ng_sp - (sig_fig+1) - 4 
            # l_sp = max(l_sp, 0)

            # return " "*ng_sp+e_formatter.format(num) + l_sp*" " *sp


            if precision < res:

                e_formatter = "{:."+str(precision)+"e}"

                l_sp = len_str - ng_sp - (sig_fig+1) - 4 
                l_sp = max(l_sp, 0)

                return " "*ng_sp+e_formatter.format(num) + l_sp*" " *sp

            else:

                e_formatter = "{:."+str(res)+"e}"

                l_sp = len_str - ng_sp - (2+res) - 4 
                l_sp = max(l_sp, 0)

            return " "*ng_sp+e_formatter.format(num) + l_sp*" " *sp


        
    return ff




def fexp(f):
    return int(floor(log10(abs(f)))) if f != 0 else -np.inf

def fman(f):
    return f/10**fexp(f)






###############################################################################
                             # EXPLORATION #
###############################################################################




def var_name(var, dict_loc) :

    if type(dict_loc) != dict :
        raise ValueError("dict_loc should be a dict")

    if var not in dict_loc.values() :
        raise ValueError("variable "+str(var)+" not in dict_loc")

    for k,v in dict_loc.items():
        if v == var :
            return(k)



def tree(object, object_name="instance of"):
    """
    print tree of object attributes and associated type
        arg 1 : object
        arg 2 : object_name under string format == " "
    """

    if type(object_name) != str :
        raise ValueError("object_name should be a string")

    if "DataFrame" in str(type(object)):
        return object

    keys = [

        k for k in dir(object) 
            if (
                "method" not in str(type(object.__getattribute__(k)))
                and "__" not in str(k)
                )                             
            ]


    
    obj = Node(object_name+" "+str(type(object))[8:-2])

    for k in keys:

        value = object.__getattribute__(k)

        str_name = k + (19-len(k))*" "+" : "

        str_type = str(type(value))[8:-2]
        if "numpy.ndarray" in str_type :
            str_name += "array               : " + str(value.shape)

        elif "list" in str_type :
            str_name += "list                : " + str(len(value))

        elif "Series" in str_type :
            str_name += "Series              : " + str(value.shape) 

        elif "DataFrame" in str_type :
            str_name += "DataFrame           : "+ str(value.shape)

        elif "dict" in str_type :
            str_name += "dict                : " + str(len(value))

        elif "float" in str_type:
            str_name += "int                 : " + str(value)

        elif "int" in str_type:
            str_name += "float               : " + str(value)

        else: 
            str_type = str_type.split(".")[-1]
            str_name += str_type

        nod = Node(str_name, parent=obj)


    print()

    for pre, fill, node in RenderTree(obj):

        print("    %s%s" % (pre, node.name))

    print()




def locals_object(dict_loc):
    """ 
    dict_local must be locals()
    """
    
    if type(dict_loc) != dict :
        raise ValueError("local should be a dict")

    keys = [

        k for k in dict_loc.keys() 

            if  str(type(dict_loc[k]))[8:-2] not in ['module','type','function']
            
            and k[0:2] != "__"

            and "builtin" not in str(type(dict_loc[k]))

            and "Wrapper" not in str(type(dict_loc[k]))

            and "ufunc" not in str(type(dict_loc[k]))
            ]



    obj = Node("objects in locals()")


    for k in keys:

        value = dict_loc[k]

        str_name = k + (19-len(k))*" "+" : "

        str_type = str(type(value))[8:-2]
        if "numpy.ndarray" in str_type :
            str_name += "array               : shape = " + str(value.shape)

        elif "list" == str_type :
            str_name += "list                : " + "("+str(len(value))+")"

        elif "Series" in str_type :
            str_name += "Series              : shape = " + str(value.shape) 

        elif "DataFrame" in str_type :
            str_name += "DataFrame           : shape = "+ str(value.shape)

        elif "dict" == str_type :
            str_name += "dict                : len   = " + str(len(value))

        elif "float" in str_type:
            str_name += "float               : " + str(value)

        elif "int" == str_type:
            str_name += "int                 : " + str(value)

        elif "str" == str_type:
            str_name += "string              : len   =" + str(len(value))

        elif "Timestamp" in str_type:
            str_name += "Timestamp           : " + str(value)

        else: 
            str_type  = str_type.split(".")[-1]
            str_name += str_type


        nod = Node(str_name, parent=obj)


    print()

    for pre, fill, node in RenderTree(obj):

        print("    %s%s" % (pre, node.name))

    print()



def arr_to_csv(arr, path="/home/d51680/array.csv",
               neg_space=True, sig_fig=3, len_str=9):
    
    if "array" not in str(type(arr)):
        raise ValueError("arr should be an array")



    df = pd.DataFrame(arr)

    df.to_csv(path,
              float_format = float_formatter(neg_space=neg_space,
                                             sig_fig=sig_fig,
                                             len_str=len_str),
              index = False,
              header = False )


    ### tag : ¬complex_arr ###




    if 'complex' in str(arr.dtype):

        lim_zero = 1e-6
        lim_inf  = 1e10


        with open(path,'w') as file:

            if len(arr.shape)==1 :

                for i in range(0,arr.shape[0]):

                    str_numb = complex_to_str(arr[i], lim_zero, lim_inf)

                    file.write(str_numb+",\n")


            elif len(arr.shape)==2:

                for i in range(0, arr.shape[0]):

                    for j in range(0,arr.shape[1]):

                        str_numb = complex_to_str(arr[i,j], lim_zero, lim_inf)

                        file.write(str_numb+',')

                    file.write('\n')



def complex_to_str(z,lim_zero,lim_inf):

    if abs(z.real) < lim_zero:
        z = 0 + z.imag*1j

    if abs(z.imag) < lim_zero:
        z = z.real + 0j



    if abs(z)      > lim_inf:

        if abs(z.real) > abs(z.imag) :

            if z.real > 0 : str_z = ' inf               '
            if z.real < 0 : str_z = '-inf               '

        else :

            if z.imag > 0 : str_z = '         +infj     '
            if z.imag < 0 : str_z = '         -infj     '

    elif z.imag >= 0 and z.real >= 0 :

        str_z = " "+"{:.2e}".format(z.real)+"+"+"{:.2e}".format(z.imag)+"j"

    elif z.imag >= 0 and z.real < 0 :

        str_z = "{:.2e}".format(z.real)+"+"+"{:.2e}".format(z.imag)+"j"

    elif z.imag < 0 and z.real >= 0 :

        str_z = " "+"{:.2e}".format(z.real)+"{:.2e}".format(z.imag)+"j"

    elif z.imag < 0 and z.real < 0 :

        str_z = "{:.2e}".format(z.real)+"{:.2e}".format(z.imag)+"j"


    str_z = str_z.replace("+0.00e+00j" ,"          ")
    str_z = str_z.replace("-0.00e+00j" ,"          ")

    str_z = str_z.replace("0.00e+00","0       ")

    str_z = str_z.replace("0       +","         ")
    str_z = str_z.replace("0       -","        -")


    return str_z




def df_to_csv(df,
              path="/home/d51680/df.csv",
              sig_fig=4, 
              len_str=10,
              resolution=4,
              align_col=False):
    


    if "DataFrame" not in str(type(df)):
        if "Series" not in str(type(df)):
            raise ValueError("df should be a DataFrame or Series")

    try:
        df = df.astype('float64')
    except:
        pass
    else:
        df = df.astype('float64')


    # --------------------------------------------------------------------------
    # replace column header by 10 str long with spaces

    if align_col:

        if len_str < 10 and sig_fig==4:
            print("WARNING: rsl.py:690 df_to_csv : len_str and sig_fig not compatible")

        col = list(df.columns)

        for i in range(len(col)):

            if len(col[i]) < len_str:

                col[i] += (len_str-len(col[i]))*' ' 


        df.columns = col


        # index_name = df.index.name

        # index = list(df.index)

        # for i in range(len(index)):

        #     if len(index[i]) < 10:

        #         index[i] += (10-len(index[i]))*' '


        # df.index = index

        # if index_name != None: 

        #     if len(index_name)< 10:

        #         df.index.name = index_name + (10-len(index_name))*' '
    
    

    df.to_csv(path,
              float_format = float_formatter(neg_space=True,
                                             sig_fig=sig_fig,
                                             resolution=resolution,
                                             len_str=len_str)      
              )



    # --------------------------------------------------------------------------
    # replace NaN with empty string 10 character long

    with open(path, 'r') as file:
        filedata = file.read()

    filedata = filedata.replace(',,', ','+10*' '+',')
    filedata = filedata.replace(',,', ','+10*' '+',')

    with open(path, 'w') as file:
        file.write(filedata)


def read_csv(path,squeeze=False, rm_space=True, dtype=None):
    
    df = pd.read_csv(path,
                     parse_dates=True,
                     squeeze=squeeze,
                     index_col=0,
                     dtype=dtype    )


    if "Serie" not in str(type(df)) and rm_space:

        df.columns = [ c.replace(' ','') for c in list(df.columns)]
   
        # if df.index.name != None:
        #     index_name = df.index.name.replace(' ','')

    # df.index   = [ i.replace(' ','') for i in list(df.index)]

    # if df.index.name != None:
    #     df.index.name = index_name



    return df


def read_arr(path):

    df = pd.read_csv(path,
                     header=None)

    array = df.to_numpy()

    if len(array.shape) == 2 and array.shape[1] ==1:

        array = array.reshape( (array.shape[0],) )

    return array



def print_log(path_log, content, mode='a'):

    if type(path_log) != str:
        raise ValueError("path_log should be a string")

    if type(content) != str :
        raise ValueError("content should be a string")

    if not os.path.exists(str):
        raise ValueError("path doesn't exist")

    with open(path_log, mode) as f:
        f.write(content)

    print(content)




def ovdir(psdx, psdy, wave_dir):

    Ax  = sqrt(psdx)
    Ay  = sqrt(psdy)

    ang = atan(Ax/Ay) * 180/pi


    if wave_dir < 60 or  330 <= wave_dir:

        vib_dir = 240 - ang


    elif 60 <= wave_dir < 150:

        vib_dir = 240 + ang
    
    elif 150 <= wave_dir < 240:

        vib_dir = 60 - ang

    elif 240 <= wave_dir < 330:

        vib_dir = 60 + ang


    opp_vib_dir = (vib_dir - 180)%360

    # print("wave_dir = ", wave_dir)
    # print("nf1 vibration direction =", vib_dir)

    # print("\nnf1 opposed vibration direction =")
    
    return opp_vib_dir



def rotdir_nf1(ASX_am, ASY_am, wave_dir):

    Ax  = ASX_am
    Ay  = ASY_am

    ang = atan(Ax/Ay) * 180/pi


    if 330 <= wave_dir or wave_dir < 60 :

        rot_dir = 150 - ang

    elif 60  <= wave_dir < 150 :

        rot_dir = 150 + ang 

    elif 150 <= wave_dir < 240 :

        rot_dir = 330 - ang

    elif 240 <= wave_dir < 330 : 

        rot_dir = ( 330 + ang ) % 360


    return rot_dir




def oadir(Xi, Yi):

    theX =  30   * pi/180
    theY =  120  * pi/180

    if Yi == 0 :

        theta = theY - np.sign(Xi)*pi/2

    if Yi  > 0 :

        theta = theY - atan(Xi/Yi)

    if Yi  < 0 : 

        theta = theX - pi/2 - atan(Xi/Yi)

    avg_dir = - theta*180/np.pi%360 


    opp_avg_dir = (avg_dir + 180)%360

    
    return opp_avg_dir






def arrow_circ_h(ax,center, radius, facecolor='#2693de', edgecolor='#000000', theta1=0, theta2=200):
   
   # Add the ring
   rwidth = 0.002
   ring = patches.Wedge(center, radius, theta1, theta2, width=rwidth)
   # Triangle edges
   offset = 0.006
   xcent  = center[0] + radius - (rwidth/2)
   left   = [xcent - offset, center[1]]
   right  = [xcent + offset, center[1]]
   bottom = [(left[0]+right[0])/2., center[1]-0.015]
   arrow  = plt.Polygon([left, right, bottom, left])
   p = PatchCollection(
       [ring, arrow], 
       edgecolor = edgecolor, 
       facecolor = facecolor,
   )
   ax.add_collection(p)


def arrow_circ_v(ax,center, radius, facecolor='#2693de', edgecolor='#000000', theta1=-30, theta2=180):
   
   # Add the ring
   rwidth = 0.002
   ring = patches.Wedge(center, radius, theta1, theta2, width=rwidth)
   # Triangle edges
   offset = 0.006
   # xcent  = center[0] - radius + (rwidth/2)
   ycent  = center[1] - radius + (rwidth/2)
   # left   = [xcent - offset, center[1]]
   left   = [center[0], ycent - offset, ]
   # right  = [xcent + offset, center[1]]
   right  = [center[0], ycent+offset]
   bottom = [ center[0]-0.015, (left[1]+right[1])/2.]
   arrow  = plt.Polygon([left, right, bottom, left])
   p = PatchCollection(
       [ring, arrow], 
       edgecolor = edgecolor, 
       facecolor = facecolor,
   )
   ax.add_collection(p)