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

// Tests muiltiple C++ functions from the svUtils file

IMPORT TextVectors.internal.svUtils AS Utils;

// Helper Functions

// For set of length 3
STRING SetToString(SET OF REAL S) := FUNCTION
    RETURN '[' + S[1] + ',' + S[2] + ',' + S[3] + ']';
END;

STRING IntResultCheck(INTEGER Expected, INTEGER Result) := FUNCTION
    RETURN IF(Expected = Result, 'Pass', 'Expected: ' + Expected + ' Result: ' + Result);
END;

STRING BoolResultCheck(BOOLEAN Expected, BOOLEAN Result) := FUNCTION
    RETURN IF(Expected = Result, 'Pass', 'Expected: ' + Expected + ' Result: ' + Result);
END;

STRING SetResultCheck(SET OF REAL Expected, SET OF REAL Result) := FUNCTION
    RETURN IF(Expected = Result, 'Pass', 'Expected: ' + SetToString(Expected) + ' Result: ' + SetToString(Result));
END;

STRING SetResultCheck2(SET OF REAL Expected, SET OF REAL Result) := FUNCTION
    RoundedResults := [ROUND(Result[1], 2), ROUND(Result[2], 2), ROUND(Result[3], 2)];
    RETURN IF(Expected = RoundedResults, 'Pass', 'Expected: ' + SetToString(Expected) + ' Result: ' + SetToString(RoundedResults));
END;

// Test 1: isNumeric

OUTPUT(BoolResultCheck(TRUE, Utils.isNumeric('3')), NAMED('Test1A'));
OUTPUT(BoolResultCheck(FALSE, Utils.isNumeric('Hello')), NAMED('Test1B'));
OUTPUT(BoolResultCheck(FALSE, Utils.isNumeric('T0day')), NAMED('Test1C'));
OUTPUT(BoolResultCheck(TRUE, Utils.isNumeric('01256')), NAMED('Test1D'));

// Test 2: addVecs

// The third paramter multiplies the second vector

OUTPUT(SetResultCheck(Utils.addVecs([1, 2, 3], [3, 4, 5]), [4, 6, 8]), NAMED('Test2A'));
OUTPUT(SetResultCheck(Utils.addVecs([4, 9, 2], [1, 1, 1], 8), [12, 17, 10]), NAMED('Test2B'));
OUTPUT(SetResultCheck(Utils.addVecs([1, 2, 3], [6, 13, 20], 0), [1, 2, 3]), NAMED('Test2C'));


// Test 3: normalizeVector

// This function returns the unit vector, so the result should be a set with each
// number divided by the magnitude

OUTPUT(SetResultCheck2([0, 0.6, 0.8], Utils.normalizeVector([0, 3, 4])), NAMED('Test3A'));
OUTPUT(SetResultCheck2([0.40, 0.56, 0.72], Utils.normalizeVector([5, 7, 9])), NAMED('Test3B'));
OUTPUT(SetResultCheck2([0.97, 0.21, 0.15], Utils.normalizeVector([100, 21.3, 15])), NAMED('Test3C'));

// Test 4: cosineSim

OUTPUT(IntResultCheck(5, Utils.cosineSim([1, 2], [1, 2], 2)), NAMED('Test4A'));
OUTPUT(IntResultCheck(0, Utils.cosineSim([8, 0], [0, 100], 2)), NAMED('Test4B'));
OUTPUT(IntResultCheck(320, Utils.cosineSim([80, 9, 2], [4, 8, 50], 1)), NAMED('Test4C')); // Due to the third parameter only the 
                                                                                     //first indexes should be multiplied
                                                                                     