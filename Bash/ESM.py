import esm
import numpy as np
import os
import pandas as pd
import pickle as pk
import sys
import time
import torch


# Get params
modelParams = sys.argv[1]
enzymeName = sys.argv[2]
fixAA = sys.argv[3]
fixPos = sys.argv[4]
lenSubs = int(sys.argv[5])
useReadingFrame = sys.argv[6]
minES = sys.argv[7]
minSubCount = sys.argv[8]
batchSize = int(sys.argv[9])
loadPredSubs = sys.argv[10].lower() == 'true'

# Parameters: Dataset
labelsXAxis = [f'R{i}' for i in range(1, lenSubs+1)]
enzyme = None
if enzymeName.lower() == 'mpro2':
    enzyme = f'SARS-CoV-2 M{'ᵖʳᵒ'}'
tagFile = f'{fixAA}@R{fixPos}'
fileName = None
if useReadingFrame:
    datasetTag = f'Reading Frame {tagFile}'
    fileName = f'fixedMotifSubs - {enzyme} - {tagFile} - FinalSort - MinCounts {minSubCount}'
else:
    datasetTag = f'Filtered {tagFile}'
    fileName = f'fixedSubs - {enzyme} - {tagFile} - FinalSort - MinCounts {minSubCount}'
if loadPredSubs == 'true':
    loadPredSubs = False if loadPredSubs.lower() == 'false' else loadPredSubs
print(f'Generate Embeddings:\n'
      f'    ESM: {modelParams}\n'
      f'    Enzyme: {enzymeName}\n'
      f'    Min Subs: {minSubCount}\n\n')


# Get: Predicted substrate file name
if loadPredSubs:
    paths = {
        '5000': "genSubs - SARS-CoV-2 Mᵖʳᵒ - Reading Frame Q@R4 - MinES 0 - "
                "Added I@R1_H@R2_[R,C,G]@R6_V@R7_[S,G,M]@R8 - MinCounts 5000",
        '1000': "genSubs - SARS-CoV-2 Mᵖʳᵒ - Reading Frame Q@R4 - MinES 0 - "
                "Added H@R2_[R,C]@R6_V@R7_[S,M]@R8 - MinCounts 1000",
        '100': "genSubs - SARS-CoV-2 Mᵖʳᵒ - Reading Frame Q@R4 - MinES 0 - "
               "Added I@R1_H@R2_[R,C]@R6_V@R7_[S,M]@R8 - MinCounts 100",
        '10': "genSubs - SARS-CoV-2 Mᵖʳᵒ - Reading Frame Q@R4 - MinES 0 - "
              "Added I@R1_H@R2_[R,C]@R6_V@R7_[S,M]@R8 - MinCounts 10"
    }
    fileNamePredSubs = f'{paths[str(minSubCount)]}.txt'


# Define: Directories
pathData = os.path.join('Data')
pathEmbeddings = os.path.join('Embeddings')
os.makedirs(pathData, exist_ok=True)
os.makedirs(pathEmbeddings, exist_ok=True)


# Set device
print('============================== Set Training Device '
      '==============================')
if torch.cuda.is_available():
    device = 'cuda:0'
    print(f'Train with Device: {device}\n'
          f'Device Name: {torch.cuda.get_device_name(device)}'
          f'\n\n')
else:
    import platform
    device = 'cpu'
    print(f'Train with Device: {device}\n'
          f'Device Name: {platform.processor()}\n\n')



# =================================== Define Functions ===================================
def loadSubs(file, tag, loadPredictedSubs=False):
    print('================================= Loading Data '
          '==================================')
    pathSubs = os.path.join(pathData, file)
    print(f'Loading Substrates: {tag}\n'
          f'     {pathSubs}\n\n')

    if loadPredictedSubs:
        # Load substrates as a list
        with open(pathSubs, 'r') as file:
            loadedSubs = [line.strip() for line in file]
            totalSubs = len(loadedSubs)
    else:
        # Load pickled substrates as a dictionary
        with open(pathSubs, 'rb') as openedFile:  # Open file
            loadedSubs = pk.load(openedFile)  # Access the data
            totalSubs = sum(loadedSubs.values())
    print(f'Total Substrates: {totalSubs:,}\n\n')

    return loadedSubs, totalSubs



def ESM(substrates, paramsESM, tagEmbeddiongs, pathSave, trainingSet=False):
    print('=========================== Generate Embeddings: ESM '
          '============================')
    # Inspect: Data type
    predictions = True
    if trainingSet:
        predictions = False
    print(f'ESM Model: {paramsESM}\n'
          f'Batch Size: {batchSize}\n')

    # Load: ESM Embeddings
    pathEmbeddings = os.path.join(pathSave, f'{tagEmbeddiongs}.csv')

    # # Generate Embeddings
    # Step 1: Convert substrates to ESM model format and generate Embeddings
    totalSubActivity = 0
    subs = []
    values = []
    if trainingSet:
        for index, (substrate, value) in enumerate(substrates.items()):
            totalSubActivity += value
            subs.append((f'Sub{index}', substrate))
            values.append(value)
    else:
        for index, substrate in enumerate(substrates):
            subs.append((f'Sub{index}', substrate))
    sampleSize = len(substrates)
    print(f'Total unique substrates: {len(substrates):,}\n'
          f'Collected substrates: {sampleSize:,}')
    if totalSubActivity != 0:
        if isinstance(totalSubActivity, float):
            print(f'Total Values: {round(totalSubActivity, 1):,}')
    print()


    # Step 2: Load the ESM model and batch converter
    if paramsESM == '15B Params':
        print(f'Loading Model: esm2_t48_15B_UR50D()\n')
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        numLayersESM = 48
    else:
        print(f'Loading Model: esm2_t36_3B_UR50D()\n')
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        numLayersESM = 36
    # print(f'Loading Model: esm2_t33_650M_UR50D\n')
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # numLayersESM = 33
    model = model.to(device)

    # Get: batch tensor
    batchConverter = alphabet.get_batch_converter()


    # Step 3: Convert substrates to ESM model format and generate Embeddings
    try:
        batchLabels, batchSubs, batchTokens = batchConverter(subs)
        batchTokensCPU = batchTokens
        batchTokens = batchTokens.to(device)

    except Exception as exc:
        print(f'ERROR: The ESM has failed to evaluate your substrates\n\n'
              f'Exception:\n{exc}\n\n'
              f'\n\nSuggestion:'
              f'     Try adding more memory:\n'
              f'          #SBATCH --mem=[80G, 100G, 120G, 140G]\n\n')
        sys.exit(1)
    print(f'Batch Tokens: {batchTokens.shape}\n'
          f'{batchTokens}\n')

    # Record tokens
    slicedTokens = pd.DataFrame(batchTokensCPU[:, 1:-1],
                                index=batchSubs,
                                columns=labelsXAxis)
    if totalSubActivity != 0:
        slicedTokens['Values'] = values
    print(f'\nSliced Tokens:\n'
          f'{slicedTokens}\n\n')

    # Generate embeddings
    allEmbeddings = []
    allValues = []
    print('Generating ESM Embeddings')
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(batchTokens), batchSize):
            start = time.time()
            batch = batchTokens[i:i + batchSize].to(device)
            result = model(batch, repr_layers=[numLayersESM], return_contacts=False)
            tokenReps = result["representations"][numLayersESM]
            seqEmbed = tokenReps[:, 0, :].cpu().numpy()
            allEmbeddings.append(seqEmbed)
            end = time.time()
            if trainingSet:
                allValues.extend(values[i:i + batchSize])

            # Clear data to help free memory
            del tokenReps, batch
            torch.cuda.empty_cache()
    end = time.time()
    runtime = end - start
    print(f'ESM Runtime: {round(runtime, 3):,} s\n\n')


    # Step 4: Extract Per-Sequence Embeddings
    embeddings = np.vstack(allEmbeddings) # Convert to numpy
    if predictions:
        data = np.hstack([embeddings])
        columns = [f'feat_{i}' for i in range(embeddings.shape[1])]
    else:
        values = np.array(allValues).reshape(-1, 1)
        data = np.hstack([embeddings, values])
        columns = [f'feat_{i}' for i in range(embeddings.shape[1])] + ['activity']

    # Process Embeddings
    subEmbeddings = pd.DataFrame(data, index=batchSubs, columns=columns)
    print(f'Substrate Embeddings:\n{subEmbeddings}\n\n')
    print(f'Saving Embeddings At:\n'
          f'     {pathEmbeddings}\n\n')
    subEmbeddings.to_csv(pathEmbeddings)



# ===================================== Run The Code =====================================
if loadPredSubs:
    # Load: Substrates
    subsPred, subsPredN = loadSubs(file=fileNamePredSubs, tag='Prediction Data',
                                   loadPredictedSubs=True)

    # Define: File tag
    tagEmbeddingsPred = (
        f'Embeddings - ESM {modelParams} - {enzymeName} - Predictions - '
        f'Min ES {minES} - MinCounts {minSubCount} - N {subsPredN} - '
        f'{len(labelsXAxis)} AA - Batch {batchSize}')

    # Generate embeddings
    ESM(substrates=subsPred, paramsESM=modelParams, pathSave=pathEmbeddings,
        tagEmbeddiongs=tagEmbeddingsPred)
else:
    # Load: Substrates
    subsTrain, subsTrainN = loadSubs(file=fileName, tag='Training Data')

    # Define: File tag
    tagEmbeddingsTrain = (
        f'Embeddings - ESM {modelParams} - {enzyme} - {datasetTag} - '
        f'MinCounts {minSubCount} - N {subsTrainN} - '
        f'{len(labelsXAxis)} AA - Batch {batchSize}')

    # Generate embeddings
    ESM(substrates=subsTrain, paramsESM=modelParams, pathSave=pathEmbeddings,
        tagEmbeddiongs=tagEmbeddingsTrain, trainingSet=True)

