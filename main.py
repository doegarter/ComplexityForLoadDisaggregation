from matplotlib.pyplot import *
import numpy
import time
import collections
import pandas
import datetime
from math import *




class Appliance(object):
    def __init__(self, *states):
        self.states = [] 
        self.state = 0
        self.p = None
        self.pdfs = []

        
        self.__add_states(states)
    
    def add_states_as_collection(self, states):
        self.__add_states(states)

    def add_states(self, *states):
        self.__add_states(states)


    def __add_states(self,states):
        for s in states:
            self.states.append(s)
            self.pdfs.append(PDF(s))

    def add_state(self, state):
        self.state = state
        self.p = PDF(state)

    def __str__(self):
        return (str(type(self))+'(states: '+str(self.states)+')')


    def get_probability_at(self, t):
        p = 0
        for pdf in self.pdfs:
            p = max(p, pdf.pdf(t))
        return p

    def get_max_power(self):
        power = 0
        for state in self.states:
            power = max(power, state)
        return power




class Complexity(object):
    def __init__(self, resolution = -1):
        self.appliances = []
        self.areas = []
        self.subComplexities = []
        self.max_power = 0
        self.pdfs = []
        self.x = None
        self.resolution = resolution
        self.variance = 5
        self.complexity_sum = 0.0
    

    def add_appliance(self, *appliances):
        for a in appliances:
            self.appliances.append(a)

    def calc_areas(self):

        ''' Calculates the areas of the interlacing pdfs
            ---> Run generate_pdfs first! '''
        print('calculating areas...')
    
        n_pdfs = len(self.all_pdfs)
        #self.areas = n_pdfs*[n_pdfs*[-1]]
        p = 0
        for i in range(n_pdfs):
            pn = (int)((i*100)/len(self.all_pdfs))
            if(p != pn):
                print(str(p)+'%')
            
            p = pn
            self.areas.append([])
            for j in range(n_pdfs):
                #print('calculating area',i,'/',j)
                pdf1 = self.all_pdfs[i]
                pdf2 = self.all_pdfs[j]
                tmp_area = Area(pdf1,pdf2,self.max_power)
                tmp_area.calc_cross_section(self.x)
                self.areas[i].append(tmp_area)
                #self.areas[i][j] = tmp_area
                if self.areas[i][j].a > 0:
                    pass#print(self.areas[i][j].a)
  
    def calc_total_complexity(self):
        print('calculating total complexity...') 
        sum = 0
        c = 0
        l = len(self.power_vals)
        print(l)
        for v in self.power_vals:
            c+=1
            print(str(int((c*100)/l))+'%')
            print('sum: '+str(sum))
            pdfk = PDF(v)
            pdfk.calc_normal(self.x, self.variance)
            for j in self.all_pdfs:
                pdfj = j
                ar = Area(pdfk, pdfj, self.max_power)
                ar.calc_cross_section(self.x)
                sum+=ar.a


        total = sum/len(self.power_vals.keys())
        print('total: '+str(total))
    
   




    def calc_subcomplexities(self):

        self.total_complexity = 0
        for a in range(len(self.all_pdfs)):
            #print('calculating complexity', a)
            sc = Subcomplexity()
            sc.calc_constant_sum(a, self.all_pdfs, self.areas)
            self.subComplexities.append(sc)
            

            self.complexity_sum += sc.value
        print('mean: '+ str(self.complexity_sum/len(self.all_pdfs)))



    def plot(self):
       ar = []
       for x in range(self.resolution):
           ar.append(0)
           for p in self.all_pdfs:
               #print(x)
               ar[x]=max(p.pdf(x), ar[x])

       plot(self.x, ar)

    def calc_max_power(self):
        self.max_power = 0
        for p in self.all_pdfs:
            self.max_power = max(self.max_power, p.loc)
 
    def generate_pdfs(self):
        tmp_pdfs = []

        self.all_pdfs = []
        for a in self.appliances:
            tmp_pdfs.append(a.pdfs)
        
        self.add_up(0,tmp_pdfs)
        print(len(self.all_pdfs))
        #for i in range(len(tmp_pdfs)):
        #    for j in range(i, len(tmp_pdfs)):
        #        p1 = tmp_pdfs[i]
        #        p2 = tmp_pdfs[j]
        #        if(p1 != p2):
        #            tmp_pdfs.append(PDF(p1.loc+p2.loc))
        #            print(p1.loc+p2.loc)
        
        
        
        self.calc_max_power()
        if self.resolution == -1:
            self.resolution = self.max_power+1
        self.x = numpy.linspace(0, self.max_power, self.resolution)
        #print(self.x)
        for p in self.all_pdfs:
            p.calc_normal(self.x, self.variance)


    def add_up(self, power, pdfs):
        for p in pdfs[0]:
            p = power+p.loc
            if len(pdfs) > 1:
                self.add_up(p, pdfs[1:])
            else:
                self.all_pdfs.append(PDF(p)) 

    def plot_complexity(self):
        y = [] 
        for c in self.subComplexities:
            y.append(c.value)

        x = []
        for p in self.all_pdfs:
            x.append(p.loc)

        plot(x,y, 'ro')

    def __print_percent(old_p, a, g):
        p = int((a*100)/g)
        if p > old_p:
            print(str(p)+'%')
        return p


    def __med(n, p):
        r = int(n/2)
        vals = []
        old_p = 0
        for i in range(len(p[0])):
            old_p = Complexity.__print_percent(old_p, i, len(p[0])) 
            s = (max(i-r,0))
            e = (min(i+r+1, len(p[0])))
            tmp_frame = ((p[0][s:e]).to_frame())
            median = tmp_frame.median()
            vals.append(median)
        return pandas.DataFrame(vals, index=[p.index])
      
        
        
    def load_dataset(self, house_n, dataset):

        #used to limit dataset
        n = 10000 

        path = 'REDD/low_freq/house_'+str(house_n)+'/'
        power_vals = {}
        for d in dataset:
            file_name = path+'channel_'+str(d)+'.dat'
            print('loading '+file_name+'...')
            f = open(file_name, 'r')
            for line in f:
                splitted_line = line.split()
                key, value = int(splitted_line[0]), float(splitted_line[1])
                #print(str(key)+' ::: '+str(key in power_vals.keys()))
                if key in power_vals.keys():
                    power_vals[key] += value
                else:
                    power_vals.update({key:value})
                    #print(str(key)+':'+str(power_vals[key]))
        print('sorting dict...')
        power_vals = collections.OrderedDict(sorted(power_vals.items()))
        print('creating pandas object...')
        tmp_list = list(power_vals.keys())
        reduced_list =  tmp_list[:n]
        timestamps = [datetime.datetime.fromtimestamp(t) for t in reduced_list]
        timestamps = list(timestamps)
        dates = pandas.DatetimeIndex(timestamps)
        print('timestamps done!')

        vals = [l[1] for l in list(power_vals.items())[:n]]
        pandaso = pandas.DataFrame(vals, index=[dates])

        


        print('resampling...')
        pandaso = pandaso.resample(rule='1S', how='mean')
        
        print(len(pandaso))
        pandaso = pandaso.fillna(method = 'ffill')
        print(len(pandaso))
        pandaso = pandaso.resample(rule='10S', how='mean')

        pandaso.plot()
        


        pandaso = Complexity.__med(19, pandaso)
        
        self.power_vals = pandaso[0]
        print(self.power_vals)
        

        pandaso.plot()




        #self.power_vals = collections.OrdereddDict()        
        #print('resizing dict...')
        #for i in range(200):
        #    key = list(power_vals.keys())[i*3500]
        #    #print(key)
        #    value = power_vals[key]
        #    self.power_vals.update({key:value})
        #plot(list(self.power_vals.keys()),list(self.power_vals.values()))

class Subcomplexity:
    def __init__(self):
        self.value = 0.0

    def calc_constant_sum(self, k, all_pdfs, super_areas):
        for a in range(len(all_pdfs)):
            ar = super_areas[a][k].a
            if(ar == -1):
                ar=super_areas[k][a].a 
                #print('IS -1!')
            self.value+=ar
            #print(str(k)+' / '+str(a)+': '+str(ar))
        #print(self.value)



class Area(object):
    def __init__(self, pdf1, pdf2, max_power):
        self.a = 0
        self.error = 0
        self.pdf1 = pdf1
        self.pdf2 = pdf2
        self.max_power = max_power

    def calc_cross_section(self, x):
        #ar = quad(self.minimum, 0, self.max_power)
        ar = numpy.trapz(self.minimum(), x)
        self.a = ar#        self.error = ar[1]
        
    def minimum(self):
        
        r = []
        for i in range(len(self.pdf1.n)):
            t1 = self.pdf1.pdf(i)
            t2 = self.pdf2.pdf(i)
            r.append(min(t1,t2))
                         
        
        return r


class PDF(object):
    def __init__(self, loc,):
        self.loc = loc
        self.n = []
        self.x = []

    def calc_normal(self, x, variance):
        self.n = []
        #self.x = x
        for i in x:
            dividend = exp(-(((i*1.0-self.loc)/variance)**2)/2.0)
            divisor =variance*sqrt(2.0*pi)
            self.n.append(dividend/divisor)
 
        #leave commented for better performance
        #print('>>>>'+str(numpy.trapz(self.n, x)))
    
    def get_loc(self):
        return self.loc

    def pdf(self, t):
        #print('           index:'+str(t))
        return self.n[t]

     
class Building(object):
    def __init__(self, n, house_n, dataset, dat):
        self.complexity = Complexity(n)
        self.appliances = []
        self.dataset = dataset 
        self.house_n = house_n
        self.dat = dat

    def add_appliances(self, *apps):
        for app in apps:
            self.appliances.append(app)       
            self.complexity.add_appliance(app)

    def load_apps(self):
        f = open(self.dat, 'r')
        
        for l in f:
            tmp = l.split(' ')
            n = int(tmp[0])
            states = tmp[1].split(',')
            
            int_states = []

            for i in states:
                int_states.append(int(i))


            a = Appliance()
            a.add_states_as_collection(int_states)
            if n in self.dataset:
                print(int_states)
                self.add_appliances(a) 
            
          

    def calc_total_complexity(self):
        self.complexity.load_dataset(self.house_n, self.dataset)
        self.complexity.generate_pdfs()
        self.complexity.calc_total_complexity()

    def calc_subcomplexities(self):
        self.complexity.generate_pdfs()
        self.complexity.calc_areas()
        self.complexity.calc_subcomplexities()
        self.complexity.plot_complexity()


      
def main():
    t1 = time.time()
    
    redd_house_1 = Building(n = 1000, house_n = 1, dataset=(3, 5, 6, 7, 11, 19), dat = 'enteryourlocation_of_building_configuration_file')
    print(redd_house_1)
    redd_house_1.load_apps()
    #redd_house_1.calc_total_complexity()
    redd_house_1.calc_subcomplexities()

    t = (int)(time.time()-t1)
    print(str((int)(t/60))+':'+str((int)(t%60)))

    savefig(str(time.time())+'.png')
    show()

if __name__ == '__main__':
    main()
