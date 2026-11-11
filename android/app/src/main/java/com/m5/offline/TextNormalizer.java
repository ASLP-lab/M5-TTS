package com.m5.offline;

import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.time.*;
import java.util.*;
import java.util.regex.*;

/** Offline speech TN. Regexes and unit vocabulary are shared with Python verbatim. */
public final class TextNormalizer {
    public static final class Result {
        public final String text, version; public final boolean changed;
        Result(String text, String original, String version) { this.text = text; this.version = version; changed = !text.equals(original); }
    }
    private static final String ZH = "零一二三四五六七八九";
    private static final String[] EN = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(" ");
    private static final String[] TENS = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    private static final String[] MONTHS = {"", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
    private final JSONObject units;
    private final List<Rule> rules = new ArrayList<>();
    public final String version;
    private static final class Rule {
        final String kind; final Pattern pattern;
        Rule(JSONObject row) throws JSONException {
            kind = row.getString("handler");
            pattern = Pattern.compile(row.getString("pattern"), row.getBoolean("ignore_case") ? Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE : 0);
        }
    }
    private static final class Part {
        final String text; final boolean protectedText;
        Part(String text, boolean value) { this.text = text; protectedText = value; }
    }
    public TextNormalizer(InputStream in) throws IOException, JSONException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream(); byte[] block = new byte[8192]; int count;
        while ((count = in.read(block)) != -1) { if (bytes.size() + count > 512*1024) throw new IOException("TN 规则文件过大"); bytes.write(block, 0, count); }
        JSONObject data = new JSONObject(new String(bytes.toByteArray(), StandardCharsets.UTF_8));
        version = data.getString("version"); units = data.getJSONObject("units");
        JSONArray rows = data.getJSONArray("rules"); for (int i=0; i<rows.length(); i++) rules.add(new Rule(rows.getJSONObject(i)));
    }
    public Result normalize(String input, boolean enabled, String mode, String language) {
        if (input == null || input.codePointCount(0, input.length()) > 8000) throw new IllegalArgumentException("TN 输入文本无效或过长");
        if (!Arrays.asList("auto", "cardinal", "digits").contains(mode) || !Arrays.asList("Chinese", "English").contains(language)) throw new IllegalArgumentException("TN 参数无效");
        if (!enabled) return new Result(input, input, version);
        boolean english = language.equals("English"); StringBuilder folded = new StringBuilder();
        for (int i=0; i<input.length(); i++) {
            char c = input.charAt(i);
            if ((c >= 0xff10 && c <= 0xff19) || (c >= 0xff21 && c <= 0xff3a) || (c >= 0xff41 && c <= 0xff5a) || "％＋－．／＝".indexOf(c)>=0) c -= 0xfee0;
            else if (c == 0x3000) c = ' '; else if (c == '−') c = '-';
            folded.append(c);
        }
        String source = folded.toString().replaceAll("(?<=[0-9])，(?=[0-9])", ",").replaceAll("(?<=[0-9])：(?=[0-9])", ":");
        List<Part> parts = new ArrayList<>(); parts.add(new Part(source, false));
        for (Rule rule : rules) {
            List<Part> next = new ArrayList<>();
            for (Part part : parts) {
                if (part.protectedText) { next.add(part); continue; }
                Matcher match = rule.pattern.matcher(part.text); int start = 0;
                while (match.find()) {
                    if (match.start() > start) next.add(new Part(part.text.substring(start, match.start()), false));
                    next.add(new Part(rewrite(rule.kind, match, english, mode), true)); start = match.end();
                }
                if (start < part.text.length()) next.add(new Part(part.text.substring(start), false));
            }
            parts = next;
        }
        StringBuilder result = new StringBuilder(); for (Part part : parts) result.append(part.text);
        String text = result.toString().replaceAll("[ \\t]+", " ").trim().replaceAll(" +([，。！？；、,.;!?])", "$1");
        return new Result(text, input, version);
    }
    private static String digits(String value, boolean english) {
        StringBuilder out = new StringBuilder();
        for (int i=0; i<value.length(); i++) { if (english && i>0) out.append(' '); out.append(english ? EN[value.charAt(i)-'0'] : String.valueOf(ZH.charAt(value.charAt(i)-'0'))); }
        return out.toString();
    }
    private static String zhGroup(int number) {
        StringBuilder out = new StringBuilder(); boolean zero = false;
        int[] divisors={1000,100,10,1}; String[] suffix={"千","百","十",""};
        for (int i=0; i<4; i++) {
            int digit=number/divisors[i]; number%=divisors[i];
            if (digit!=0) { if (zero) out.append('零'); out.append(ZH.charAt(digit)).append(suffix[i]); zero=false; }
            else if (out.length()>0 && number!=0) zero=true;
        }
        return out.toString();
    }
    private static String enGroup(int number) {
        List<String> out=new ArrayList<>();
        if (number>=100) { out.add(EN[number/100]); out.add("hundred"); number%=100; }
        if (number>=20) { out.add(TENS[number/10]); number%=10; }
        if (number>0) out.add(EN[number]); return String.join(" ",out);
    }
    private static String cardinal(String raw, boolean english) {
        String value=raw.replaceFirst("^0+(?!$)", "");
        if (value.length()>16) return digits(value,english);
        long number=Long.parseLong(value); if (number==0) return english?"zero":"零";
        int base=english?1000:10000; List<Integer> groups=new ArrayList<>();
        while (number>0) { groups.add((int)(number%base)); number/=base; }
        if (english) {
            String[] scales={"","thousand","million","billion","trillion","quadrillion"}; List<String> out=new ArrayList<>();
            for (int i=groups.size()-1;i>=0;i--) if(groups.get(i)!=0) out.add((enGroup(groups.get(i))+" "+scales[i]).trim());
            return String.join(" ",out);
        }
        String[] scales={"","万","亿","万亿"}; StringBuilder out=new StringBuilder(); boolean zero=false;
        for(int i=groups.size()-1;i>=0;i--) {
            int group=groups.get(i); if(group==0) { if(out.length()>0) zero=true; continue; }
            if(out.length()>0 && (zero||group<1000)) out.append('零');
            out.append(zhGroup(group)).append(scales[i]); zero=false;
        }
        return out.indexOf("一十")==0?out.substring(1):out.toString();
    }
    private static String number(String raw, boolean english) { return number(raw,english,"cardinal"); }
    private static String number(String raw, boolean english, String mode) {
        String value=raw.replace(",", ""), sign="";
        if(value.startsWith("-")||value.startsWith("+")) { boolean negative=value.charAt(0)=='-'; sign=english?(negative?"negative ":"positive "):(negative?"负":"正"); value=value.substring(1); }
        String[] parts=value.split("\\.",-1); String whole=parts[0].isEmpty()?"0":parts[0];
        boolean byDigit=mode.equals("digits")||(mode.equals("auto")&&whole.length()>1&&whole.startsWith("0"));
        String out=byDigit?digits(whole,english):cardinal(whole,english);
        if(parts.length>1) out+=(english?" point ":"点")+digits(parts[1],english);
        return sign+out;
    }
    private static String ordinal(int day) {
        switch(day) { case 1:return "first"; case 2:return "second"; case 3:return "third"; case 5:return "fifth"; case 8:return "eighth"; case 9:return "ninth"; case 12:return "twelfth"; case 20:return "twentieth"; case 30:return "thirtieth"; }
        if(day<20) return cardinal(String.valueOf(day),true)+"th";
        return TENS[day/10]+" "+ordinal(day%10);
    }
    private String unit(String key,String value,boolean english) {
        JSONObject item=units.optJSONObject(key); if(item==null)return key;
        if(!english)return item.optString("zh");
        return item.optString(Math.abs(Double.parseDouble(value.replace(",","")))==1?"en_one":"en");
    }
    private static String date(Integer year,int month,int day,boolean english) {
        LocalDate.of(year==null?2000:year,month,day);
        if(english)return MONTHS[month]+" "+ordinal(day)+(year==null?"":", "+cardinal(String.valueOf(year),true));
        return (year==null?"":digits(String.valueOf(year),false)+"年")+cardinal(String.valueOf(month),false)+"月"+cardinal(String.valueOf(day),false)+"日";
    }
    private String rewrite(String kind,Matcher m,boolean english,String mode) {
        String raw=m.group();
        switch(kind) {
            case "keep":return raw;
            case "date":
                try { if(Integer.parseInt(m.group(1))==0)return raw; return date(Integer.valueOf(m.group(1)),Integer.parseInt(m.group(2)),Integer.parseInt(m.group(3)),english); }
                catch(DateTimeException e) { return raw; }
            case "month_day":
                try { return date(null,Integer.parseInt(m.group(1)),Integer.parseInt(m.group(2)),english); }
                catch(DateTimeException e) { return raw; }
            case "year_month": {
                int month=Integer.parseInt(m.group(2)); if(month<1||month>12)return raw;
                return english?MONTHS[month]+" "+cardinal(m.group(1),true):digits(m.group(1),false)+"年"+cardinal(m.group(2),false)+"月";
            }
            case "year":return english?cardinal(m.group(1),true):digits(m.group(1),false)+"年";
            case "time": {
                int hour=Integer.parseInt(m.group(1)),minute=Integer.parseInt(m.group(2)); Integer second=m.group(3)==null?null:Integer.valueOf(m.group(3));
                if(hour>23||minute>59||(second!=null&&second>59))return raw;
                String out;
                if(english) {
                    out=cardinal(String.valueOf(hour),true);
                    if(minute>0)out+=" "+(minute<10?"oh ":"")+cardinal(String.valueOf(minute),true);
                    else out+=second==null?" o'clock":" zero";
                    return out+(second==null?"":" and "+cardinal(String.valueOf(second),true)+" seconds");
                }
                out=(hour==2?"两":cardinal(String.valueOf(hour),false))+"点";
                if(second==null&&minute==0)return out+"整";
                if(minute<10)out+="零"; if(minute>0)out+=cardinal(String.valueOf(minute),false);
                return out+"分"+(second==null?"":cardinal(String.valueOf(second),false)+"秒");
            }
            case "phone": {
                List<String> parts=new ArrayList<>(); for(String part:m.group(1).split("-"))parts.add(digits(part.replace(" ",""),english));
                return String.join(english?" ":"，",parts);
            }
            case "context_digits": {
                String separator=m.group(2),label=m.group(1); char last=label.charAt(label.length()-1);
                if(separator.isEmpty()&&((last>='A'&&last<='Z')||(last>='a'&&last<='z')))separator=" ";
                return label+separator+number(m.group(3),english,mode.equals("cardinal")?"cardinal":"digits");
            }
            case "ip": {
                List<String> groups=new ArrayList<>(); for(int i=1;i<=4;i++) { if(Integer.parseInt(m.group(i))>255)return raw; groups.add(digits(m.group(i),english)); }
                return String.join(english?" dot ":"点",groups);
            }
            case "version": {
                String value=m.group(1)!=null?m.group(1):m.group(2); List<String> parts=new ArrayList<>();
                for(String part:value.split("\\."))parts.add(number(part,english,"auto"));
                return (m.group(1)==null?"":english?"version ":"版本")+String.join(english?" point ":"点",parts);
            }
            case "range":return number(m.group(1),english)+(english?" to ":"到")+number(m.group(2),english)+(english?" ":"")+unit(m.group(3),m.group(2),english);
            case "currency": {
                String currency=m.group(1);
                if(currency.equals("¥")||currency.equals("￥"))currency="CNY";
                else if(currency.equals("$"))currency="USD"; else if(currency.equals("€"))currency="EUR"; else if(currency.equals("£"))currency="GBP";
                return number(m.group(2),english)+(english?" ":"")+unit(currency,m.group(2),english);
            }
            case "percent":return english?number(m.group(1),true)+(m.group(2).equals("%")?" percent":" per mille"):(m.group(2).equals("%")?"百分之":"千分之")+number(m.group(1),false);
            case "fraction":
                if(m.group(2).replace("0", "").isEmpty())return raw;
                return english?number(m.group(1),true)+" over "+number(m.group(2),true):number(m.group(2),false)+"分之"+number(m.group(1),false);
            case "measurement":return number(m.group(1),english)+(english?" ":"")+unit(m.group(2),m.group(1),english);
            case "scientific":return number(m.group(1),english)+(english?" times ten to the power of ":"乘以十的")+number(m.group(2),english)+(english?"":"次方");
            case "acronym":return " "+String.join(" ",m.group(1).split(""))+" ";
            case "number":return number(m.group(1),english,mode);
            default:throw new IllegalArgumentException("Unknown TN rule: "+kind);
        }
    }
}
