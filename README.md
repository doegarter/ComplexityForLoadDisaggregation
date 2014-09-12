ComplexityForLoadDisaggregation
===============================

CREATING A BUILDING OBJECT
When creating a Building object it is possible to give the number of steps of the combinations to the constructor. The number of the house is also specified in the constructor as the second parameter. The third parameter is a list containing the appliances which should be used. The fourth and last parameter is a string containing the path to the file in which the states of the appliances are stored. At the moment, it only works with appliances which are within the REDD. However, parsers or converters for other datasets could be easily developed.

 ```
redd_house_1 = Building(n = 10000, house_n =1, dataset=(3, 5, 6, 7, 11, 19), dat =’/home/student/Desktop/building_1.dat’)
 ```

Afterwards, it is also important to load the appliance data into the
program. This is done using the load_apps() function.

``` 
redd_house_1.load_apps()
```
CALCULATING THE COMPLEXITY
It is possible to calculate the total complexity of the given dataset or just the individal complexities for the single appliances. These two methods of calculating the complexities are applied completely separately. The resolution which is used while calculating the total complexity is taken from the Pandas object which is created while loading the datasets. 

Total Complexity:
To calculate the total complexity of a Building object it is important to call the function calc_total_complexity() of the Building object. This loads the specified dataset and, does some resampling and applies a median filter afterwards. As the next point the PDFs are generated. The calculation of the complexity is the final step of this function.

```    
redd_house_1.calc_total_complexity()
```

Complexities of Single Appliances:
For calculating the complexities of single appliances, the measured power over time is not used. Instead, all states of all appliances must be given to the program. To calculate and plot the complex- ities of every possible state of the whole combination the function calc_subcomplexities() of the particular Building object is used. This generates all PDFs and calculates all cross-section-areas. Afterwards it creates SubComplexity a object for every possible state. At the same time the mean complexity is being calculated and printed to the command line output.
```
redd_house_3.calc_subcomplexities()
```