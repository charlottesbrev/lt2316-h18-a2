# This is the main training script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
from argparse import ArgumentParser
from keras.layers import Input, Conv2D, Dense, Activation, Flatten, Dropout
from keras import Model
import mycoco

# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA(categories):
    # building a nn

    # 0) 'input_1': no weights
    #    output: 200, 200, 3
    inputlayer = Input(shape=(200, 200, 3))

    # 1) 'conv2d_1':
    #   weights[2] {
    #       5 : {
    #           5 : { 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}}
    #           5 : { 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}}
    #           5 : { 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}}
    #           5 : { 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}}
    #           5 : { 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}, 3 : {10,10,10}}
    #       }
    #       10 (floats)
    #   }
    #    output: 195, 195, 3
    conv2dlayer = Conv2D(10, (5,5))(inputlayer)

    # 2) 'flatten_1': no weights
    #    output: ?, ?
    flattenlayer = Flatten()(conv2dlayer)

    # 3) 'activation_1': no weights
    #    output: ?, ?
    relulayer = Activation('tanh')(flattenlayer)

    #dropoutlayer = Dropout(0.1)(relulayer)
    #denseinitial = Dense(100, activation="tanh")(flattenlayer)

    # 4) 'dense_2':
    #   weights[2] {
    #       384160,  (= 10 * 196 * 196)
    #       1 (array)
    #   }
    #    output: ?, 1
    denselayer = Dense(1)(relulayer)

    # 'activation_2' layer 5: no weights
    sigmoidlayer = Activation('sigmoid')(denselayer)

    model = Model(inputlayer, sigmoidlayer)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #len(model.layers[1].get_weights()) = 2
    #len(model.layers[1].get_weights()[0]) = 5
    #len(model.layers[1].get_weights()[1]) = 10
    #len(model.layers[1].get_weights()[1][0]) = INVALID (float)

    #len(model.layers[1].get_weights()[0][0]) = 5
    #len(model.layers[1].get_weights()[0][0][0]) = 3
    #len(model.layers[1].get_weights()[0][0][0][0]) = 10

    #len(model.layers[4].get_weights()) = 2
    #len(model.layers[1].get_weights()[0]) = 384160
    #len(model.layers[1].get_weights()[1]) = 1

    # how should I get an image from this???

    categories_fixed = [[x] for x in categories]

    mycoco.setmode('train')

    # train the nn
    subcat1_ids, subcat2_ids = mycoco.query(categories_fixed)
    if len(subcat1_ids) == 0:
        print("unable to find resources for %s" % (categories[0]))
    else:
        print("found %d entries for %s" % (len(subcat1_ids), categories[0]))
    if len(subcat2_ids) == 0:
        print("unable to find resources for %s" % (categories[1]))
    else:
        print("found %d entries for %s" % (len(subcat2_ids), categories[1]))
    if len(subcat1_ids) == 0 or len(subcat2_ids) == 0:
        return

    train_images = mycoco.iter_images([subcat1_ids, subcat2_ids], [0, 1], batch=10)
    #model.fit_generator(train_images, steps_per_epoch=40, epochs=30)
    model.fit_generator(train_images, steps_per_epoch=4, epochs=3)

    mycoco.setmode('test')

    # evaluate the nn
    test_subcat1_ids, test_subcat2_ids = mycoco.query(categories_fixed)
    test_catdog_images = mycoco.iter_images([test_subcat1_ids, test_subcat2_ids], [0, 1], batch=200)
    test_imgs = next(test_catdog_images)

    predictions = model.predict(test_imgs[0])
    classes = [(1 if x >= 0.5 else 0) for x in predictions]
    correct = [x[0] == x[1] for x in zip(classes,test_imgs[1])]
    s = sum(correct)
    n = len(correct)
    print("correct %d / %d (%.1f%%)" % (s, n, 100.0*s/n) )
    #print("Option A not implemented!")

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB():
    mycoco.setmode('train')
    print("Option B not implemented!")

# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Train a model.")    
    # Add your own options as flags HERE as necessary (and some will be necessary!).
    # You shouldn't touch the arguments below.
    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('checkpointdir', type=str,
                        help="directory for storing checkpointed models and other metadata (recommended to create a directory under /scratch/)")
    parser.add_argument('modelfile', type=str, help="output model file")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
    print("Working directory at " + args.checkpointdir)
    print("Maximum instances is " + str(args.maxinstances))

    if len(args.categories) < 2:
        print("Too few categories (<2).")
        exit(0)

    print("The queried COCO categories are:")
    for c in args.categories:
        print("\t" + c)

    print("Executing option " + args.option)
    if args.option == 'A':
        optA(args.categories)
    elif args.option == 'B':
        optB()
    else:
        print("Option does not exist.")
        exit(0)
