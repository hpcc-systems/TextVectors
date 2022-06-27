/*##############################################################################

    HPCC SYSTEMS software Copyright (C) 2022 HPCC Systems®.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
############################################################################## */

// Tests different operations and fields of the weights file

IMPORT TextVectors.Types;
IMPORT TextVectors.Internal;
IMPORT Python3;

wIndex := Types.wIndex;
Slice := Types.Slice;
SliceExt := Types.SliceExt;

// Helper functions

STRING wIndexToString(wIndex FlatIndex) := FUNCTION
    RETURN 'i: ' + FlatIndex.i + ' j: ' + FlatIndex.j + ' l: ' + FlatIndex.l;
END;

STRING IntResultCheck(INTEGER Expected, INTEGER Result) := FUNCTION
    RETURN IF(Expected = Result, 'Pass', 'Expected: ' + Expected + ' Result: ' + Result);
END;

STRING WIndexResultCheck(wIndex Expected, wIndex Result) := FUNCTION
    Pass := Expected.i = Result.i AND Expected.j = Result.j AND Expected.l = Result.l;
    RETURN IF(Pass, 'Pass', 'Expected: ' + wIndexToString(Expected) + ' Result: ' + wIndexToString(Result));
END;

BOOLEAN IsAscending(DATASET(SliceExt) slices) := EMBED(Python3)
    last = -1
    lastNodeID = 0
    for slice in slices:
        if slice.nodeid != lastNodeID:
            lastNodeID = slice.nodeid
            last = -1
        if slice.sliceid < last:
            return False
        else: 
            last = slice.sliceid
    return True
ENDEMBED;

Weights := Internal.Weights([4, 5, 6]);

// nWeights should be (weights[1] * weights[2]) + (weights[2] * weights[3])
ExpectednWeights := 50;
OUTPUT(IntResultCheck(ExpectednWeights, Weights.nWeights), NAMED('Test1'));

// Test converting to and from flat index
ExpectedIndx := 39;
OUTPUT(IntResultCheck(ExpectedIndx, Weights.toFlatIndex(2, 2, 3)), NAMED('Test2'));

ExpectedRow := ROW({2, 2, 3}, wIndex);
OUTPUT(WIndexResultCheck(ExpectedRow, Weights.fromFlatIndex(39)), NAMED('Test3'));

Slices := DATASET([{1, [1.343, 2.4342, 2.4343]}, {0, [34.3, 234.232, 5.5]},
                   {3, [5.33, 9.1, 7.33]}, {2, [8.232, 2.3233, 8.632]}], Slice);

// Test if slices are sorted by sliceid
ExtSlices := Weights.toSliceExt(Slices);
OUTPUT(IF(isAscending(ExtSlices), 'Pass', 'Slices are not sorted by sliceid'), NAMED('Test4'));

// Test that all slices are concatenated
Linear := Weights.slices2Linear(ExtSlices);
NumSlices := COUNT(Slices);
NumLinWeights := COUNT(Linear.weights);
Pass := NumLinWeights = (NumSlices * COUNT(Slices[1].weights));
OUTPUT(IF(Pass, 'Pass', 'Slices are not concatenated'), NAMED('Test5'));
