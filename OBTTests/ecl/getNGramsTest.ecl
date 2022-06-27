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

// Tests the getNGrams function

IMPORT TextVectors.Types;
IMPORT TextVectors.Internal;
IMPORT Python3;

// Helper Functions

STRING SetToString(SET OF STRING s) := EMBED(Python3)
    return str(s)
ENDEMBED;

STRING GetLastElement(SET OF STRING s) := EMBED(Python3)
    return s[len(s) - 1];
ENDEMBED;

STRING SetResultCheck(SET OF STRING Expected, SET OF STRING Result) := FUNCTION
    RETURN IF(Expected = Result, 'Pass', 'Expected: ' + SetToString(Expected) + ' Result: ' + SetToString(Result));
END;

STRING StringResultCheck(STRING Expected, SET OF STRING S) := FUNCTION
    LastElement := GetLastElement(s);
    RETURN IF(Expected = LastElement, 'Pass', 'Expected: ' + Expected + ' Result: ' + LastElement);
END;

Corpus := Internal.Corpus();

StringSet1 := ['Today', 'Is', 'Tuesday', '!'];

Result1 := Corpus.getNGrams(StringSet1, 2);
Expected1 := ['_Today_Is','_Is_Tuesday','_Tuesday_!'];

OUTPUT(SetResultCheck(Expected1, Result1), NAMED('Test1'));

Result2 := Corpus.getNGrams(StringSet1, 4);
Expected2 := '_Today_Is_Tuesday_!';

// Since ngrams is equal to the length of the set, the
// last string in the set should be a concatentation
// of all of the strings
OUTPUT(StringResultCheck(Expected2, Result2), NAMED('Test2'));

StringSet2 := ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'];

Result3 := Corpus.getNGrams(StringSet2, 2);
Expected3 := ['_1_2', '_2_3', '_3_4', '_4_5', '_5_6', '_6_7', '_7_8', '_9_10'];

OUTPUT(SetResultCheck(Expected3, Result3), NAMED('Test3'));

Result4 := Corpus.getNGrams(StringSet2, 10);
Expected4 := '_1_2_3_4_5_6_7_8_9_10';

OUTPUT(StringResultCheck(Expected4, Result4), NAMED('Test4'));
