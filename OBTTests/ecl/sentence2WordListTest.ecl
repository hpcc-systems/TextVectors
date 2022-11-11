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

// Tests the sent2worldlist function

IMPORT TextVectors.Types;
IMPORT TextVectors.Internal;
IMPORT Python3;

Sentence := Types.Sentence;

// Helper Functions

STRING SetToString(SET OF STRING s) := EMBED(Python3)
    return str(s)
ENDEMBED;

STRING CheckResult(SET OF STRING Expected, SET OF STRING Result) := FUNCTION
    RETURN IF(Expected = Result, 'Pass', 'Expected: ' + SetToString(Expected) + ' Result: ' + SetToString(Result));
END;

Corpus := Internal.Corpus();

sentences := DATASET([{1, 'HPCC Systems'},
                      {2, '!have a^great:day!'},
                      {3, 'the<state>of<florida'}], Sentence);

WordList := Corpus.sent2wordList(sentences);

// Tests that everything is set to lowercase
Expected1 := ['hpcc', 'systems'];
OUTPUT(CheckResult(Expected1, WordList[1].words), NAMED('Test1'));

// Tests that certain symbols are removed
Expected2 := ['have', 'a', 'great', 'day'];
OUTPUT(CheckResult(Expected2, WordList[2].words), NAMED('Test2'));

// Tests that < and > are always treated as their own words
Expected3 := ['the', '<', 'state', '>', 'of', '<', 'florida'];
OUTPUT(CheckResult(Expected3, wordList[3].words), NAMED('Test3'));
