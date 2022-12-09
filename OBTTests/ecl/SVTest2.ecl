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

#ONWARNING(1059, ignore);

// Test that calls functions from the SentenceVectors
// file and tests aspects of their results

// Note: Some functions were not able to be tested
// due to inconsistent results

IMPORT TextVectors AS TV;
IMPORT TextVectors.Types;
IMPORT Python3;

Sentence := Types.Sentence;
Word := Types.Word;
WordInfo := Types.WordInfo;
Close := Types.Closest;
t_ModRecType := Types.t_ModRecType;

// Helper functions

STRING CheckClosest(SET OF STRING expected, DATASET(Close) sentences) := EMBED(Python3)
    i = 0
    res = ""

    for s in sentences:
      sen = str(s.closest)[2:-2]
      if expected[i] != sen:
        res += "Expected: " + expected[i] + " Result: " + sen + " "
      i += 1
  
    return res
ENDEMBED;

SV := TV.SentenceVectors(minOccurs := 1);

Sentences := DATASET([{1, 'Greetings everyone'},
                      {2, 'The [sky] is blue'},
                      {3, 'The house has four ^rooms'},
                      {4, 'HPCC-Systems programming uses ecl'}], Sentence);

Model := SV.GetModel(sentences);

// Each word should be present only once in the model,
// so count of the word 'the' should return 1 even though
// it appears twice throughout all of the sentences

// Exclude whole sentences
ModelTyp1 := Model(typ = t_ModRecType.word);
CountThe := COUNT(ModelTyp1(text = 'the'));

OUTPUT(IF(CountThe = 1, 'Pass', 'Words are counted more than once'), NAMED('Test1'));

Words := DATASET([{1, 'The'}, {2, 'house'}, {3, 'has'}, {4, 'rooms'}, {5, 'zoo'}], Word);

//The Vectors record sequence is independent from the Words sequence, so need to sort 
Vectors := SORT(SV.GetWordVectors(Model, Words), wordid);

// The last vector word is not present in the model so it should have an empty vec field
OUTPUT(IF(Vectors[5].vec = [], 'Pass', 'This word is not present but the vec field is not empty'), NAMED('Test2'));

SimilarSentences := DATASET([ {1, 'The sky is orange'},
                              {2, 'The building has nine rooms'},
                              {3, 'Goodbye everyone'},
                              {4, 'HPCC-Systems Programming uses Java'}
                            ], Sentence);

Closest := SORT(SV.ClosestSentences(Model, SimilarSentences), id);

// The correct order of sentences from model for it to be the closest sentences
ExpectedSentences := ['The [sky] is blue', 'The house has four ^rooms',
                      'Greetings everyone', 'HPCC-Systems programming uses ecl'];

ClosestResult := CheckClosest(ExpectedSentences, Closest);

OUTPUT(IF(ClosestResult = '', 'Pass', ClosestResult), NAMED('Test3'));
