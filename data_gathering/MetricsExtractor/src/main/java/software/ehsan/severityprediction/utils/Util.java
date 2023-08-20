package software.ehsan.severityprediction.utils;

public class Util {

    //source: https://www.programcreek.com/2011/04/a-method-to-detect-if-string-contains-1-uppercase-letter-in-java/

    public static boolean isAllUpperCase(String str) {
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (c >= 97 && c <= 122) {
                return false;
            }
        }
        //str.charAt(index)
        return true;
    }

    //source https://stackoverflow.com/questions/767759/occurrences-of-substring-in-a-string
    public static int countOccurence(String str, String findStr) {
        int lastIndex = 0;
        int count = 0;

        while (lastIndex != -1) {

            lastIndex = str.indexOf(findStr, lastIndex);

            if (lastIndex != -1) {
                count++;
                lastIndex += findStr.length();
            }
        }
        return count;
    }
}