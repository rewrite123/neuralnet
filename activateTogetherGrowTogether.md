# Activate Together Grow Together

A neural net should have mechanisms to grow or shrink layers and to add or remove layers. This way, the model can optimize itself for growth instead of relying on arbitrary settings humans set. Broadly speaking, layers with lots of high activation need more neurons, and layers with little to no activation need less. To determine how many neurons to add to a hidden layer, we look at the amount of chaos in that layer's weights and multiply it by a maximum number of neurons we want to allow a layer to grow by. Memory bank layers will grow slightly differently, just by using the chaos in the layer -> high usage means growth. Layers can be added to the model by looking at the difference in the preceding layer's size. If layer 5 is growthNewLayerThreshhold times bigger than layer 4, a new layer between 4 and 5 that is in between 4 and 5's size will be added that will initially pass on the values from 4 to 5 with little to no changes in the output weight - the weights will then be optimized through training.

An example growthTriggerThreshhold might be 0.01, or, 1%.

Using the methods above, I think we can let our model grow itself to an optimized size if we start out small.

shrinkTriggerThreshhold: The minimum change in held-out correctness between validation checks. ATGT counts a plateau or decline when the change is at or below this value; it does not shrink merely because cumulative correctness is low.
shrinkEpochThrshhold: The number of consecutive validation checks whose correctness change stays at or below shrinkTriggerThreshhold before ATGT attempts a shrink
growthTriggerThreshhold: The threshhold for the % of correctness the model needs to trigger a growth
growthEpochThrshhold: The number of epochs we have to stay over growthTriggerThreshhold to trigger a growth
checkpoint: The last value of the weights we had after a growth
growthFactor: The number between 1 and 0 representing the area in the layer with the highest likelyhood of benefitting from growth
growthAmount: The amount of growth needed in an area ranging from 1 to 0
growthLocation: The location new neurons will be injected into, from 0 to layer.length
growthDisparity: The different in the layer's size from it's neighbors, not including the input and output layers
growthNewLayerThreshhold: A threshhold for new layer injection. When a layer is this times smaller than it's neighbor, it will trigger a new layer growth. Input and output layers not included.
bankGrowthThreshhold: Memory bank's ability to trigger a growth, which is triggered when there is a high amount of chaos in the bank layer
bankGrowthFactor: How much the memory bank will grow it's neurons

