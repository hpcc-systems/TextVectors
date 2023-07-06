/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT Std;
EXPORT Bundle := MODULE(Std.BundleBase)
 EXPORT Name := 'TextVectors';
 EXPORT Description := 'Text Vectorization for words and sentences';
 EXPORT Authors := ['HPCCSystems'];
 EXPORT License := 'http://www.apache.org/licenses/LICENSE-2.0';
 EXPORT Copyright := 'Copyright (C) 2019 HPCC Systems';
 EXPORT DependsOn := ['ML_Core 3.2.0'];
 EXPORT Version := '1.0.1';
 EXPORT PlatformVersion := '7.0.0';
END;
