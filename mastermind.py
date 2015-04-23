"""
Mastermind solver

author: Pierre Thibault
date: March 2015
"""

import numpy as np
import time
import multiprocessing as mp
import zlib

# Pre-computed best moves for (6,10)
FIRST_MOVE = ([0, 0, 1, 2, 3, 4], 3.358910)
NEXT_MOVES = eval(zlib.decompress('x\xda\x8d\\\xe1\xce])\x8e|\x95\xfc\x9c\x91\xa2\x08\x1bl\xcc\xbe\xcaj\x9fd\x94w\xdf\xf2\x01|\x0c\xf7&==\xddQ4!|\x1c\xb0\xcbU\xb6\xe1?\xff\xa2\x9f?\xea\xbf\xff\xe7\xc7\xbf\xfe\xb7\xfc\xfc\xd1~\xfe\xd0\xe7\xd7\xf1\xf3G\xf9?\xfc\xc1\xaf6\xa8\x8dJ\xaa\xc3~\xfe\xf8\xcf\xbf0\x86\x9e\xc1\xfe\xb7~\xfe\x90\x9f?\x18#\xf1\xfbg\xb0\xf4\xa2D\x85k\xa9\xff\xfe\xf9\xc3g\xe6=3\xa5\xc9\xe7`\x15\xa9\x98\xbcjU\x1f\\\xd3\xe0\xb9\x00}f\xa69\xb3r1+\xbd[_3\xc7\x9a\xc7\x9c\xf3\xf9u\x0e\xee"$\xcc\xdc\xa8\xad\x99)\xcd\\\x9eecr\x9e\xcb(\xb5\x8f2\xc4\xecY\xb3\x7f\xce\x1e\x8c\xdf\xdb\xf3W\xf0\x9b\xfe\x0c6\xee\xad\x0e\x0c\xef\xb6f\x8e\xc1\xb2\xbf\x8e\xe2\x03\x8d\xaa\t\xbe\xcfl\xcd\x9c?p\xae9\x96\xd1{k\xd6\xa4\xb6\xba?\xb0=\x83\xe7\x82\xe7\xb4\xbc\x0fE\xfa\x10-\xdcu\xceL\xe9\x03\xe5\x19o\xcf\xe05s\x93Q\xb1h\xee\xe4\x83\xf1\xa7\xbaO\xb0?\xc3\xe4Y\xc9\x9cY\xdb0\xa5\xc6c\xac\xb1\xf2\x8c\xe5i\x11\xfbL\xda\x1c;F\'\xd1Z\x0b\xaf\xef\xcb\xdb\\\x9f\xe9\xdf\xef+\x05v$<\xf8\x19,i\xe7\xeaet\xfc\x8b|\xe2\xd1\xb96^\xcb\x98\x9b\xa1{3\xe6i\xcf\x99M\xb9\x17a#\xd6\xb5\x19e\x7f\xdf\xdc\x86\xfe\xc7\xef\xe3\xd3\xf4\xe7\'\xe2\xff\xd49\xb6\x16-\x85\n\xa9\xacU\xd4=q\x8c\x0fk\xee\xa4\xc4d\\\xeas$-\xf9I\x9c7\xedU\xd4^+\xfbY\x97\xbd\xcd\xd32\xfa3xz\x8b/\x7f.C\x07\x8d\xda\x04\x9e\xb8f.{\xf0\xf4\xa8i\x19c\xbak\xebR\x8d\x99\x8a\xfd\xfb\xf7m\xa2s\xac<\x7f\xaf/C\xd2f6\xf0\x97\xd4\xbd;\xdcJ\xf7*\xc6\xb4\x90\xf9\x89\x95\xb1\xd7R\xda4\xa4\x98\xb9N\x04x6\xda\xf6\xcc\xdaz3x M\x87\xe5s?l[\xddX&\x8am\x86S\xd1\xb1\xcf\xfa\xfcxyF\xd7}\xdabDMe4\xaa\xc7n\xe8\x9cn;\xed\\D5\xac\x19\xc7-a\x1a\x9c|[\x9e\xf1\xb6\xcd\xb9[c\x1a\xdam\xdaFM\xb6\xd1\xf7\xc6\xe9\x1cL\xbf\xca\x8d\x15\x13\x88\xf2\xb9\x8dRT\x88\x07,y\xfdp9\xf7\xb6\xcf\xb3x\xe6\x93\xdb)\xe2\xac\xea\xfc\x98\xf7Gr\x02>\xdb\x13-\x8f$\x93^\xa5s\xa1\xc3\x08K\xb2\x13\xd9\xf3Y\x1f\x03\xee\xd0\xb0G\x07\x88\xb4=mK\'d\x85\xadU\xa1^\xc7\r\xd6\xd3w--\x03\x11C\xd9\x0c\xa6\xa5\xb7\xfb\x06*\xc4\xccU\x15p\x06\x90\x9c\xbeN\xa7\xaf\xd3\xde\xfc\xe9e\xd4\x8d\xb5\x17 \xb0\xdc\xfe;Os\xee\xe94\x15\xa0G\xaf\xea\xd1\xa8\xad\xdd\xe0\x8f\xdd]k.sw\xe9<P\xdb{\xbc\x80\xa6\xab\xc2L\x93\x9d\xc6\x86\x85\xdd\x855\x89\xef\x18\xe2\xdb\x98\xbb\x1b1(0f\x1e\xedD\x1a\x19J\xd2{\x07<\x1ex`{\xf0H\xb0\x04\xdc R\xabE\x0e8\x1f\xcf|\xb4\x87\x87\r\x1a\xf0\x9f\xa8\x9ex.\xfb\xc0&<\xca\x83\xba\x02\xf3g+\xb0\x88\r\x8e\xedt\x00M\x06;\xd2\xe0\xdf7\xfc\xdb\xc6\xd2\xf8Bm\xc4\xd8\x92\xde\xb4\x1d@\x13Q\xa5\'\x1f\xef\x03K6\xa1V\xea\r4\xb6#\\\x18\x9b\x91\xb56\x8a\x1b\xe7\x97e\xd4\xd8\xeb\xcd\x0c\xaa\xd5V\x96\x83d\xa8i\xdb\xe6G\x0c\x06\xd0Y\x03\x92N\x07n\xe72^ \x9b\xa7R\x14\xa8>l\xd0\x07\xf5\x19\x9bs\xa45\x17\x1dB\x95z\xbb\xa1\xc6\x9e\xdd\x18\x11\xb2\xe0\xa7\x83\x9cU\x11\x97~\x80N\xdf\x07>\xf7Y\xf6\x07\x1a\xa2a\xeb\xda\x0f\xd0\tn7\xe2\x03\xf9\x17;-\x00\xf9\x01\xf2\x1d\xf0\xd3\xf7\x02\xda\x04\xbdg0f.\x08\x174F;\x80\xa8\xef\x00g\xe9\xb8{E4\x86m\xc0\xa8o \xb2\x14-\xe6>\xc3\x92\xaa2i\t\x04\xa0M:x\xe2i\xfe\xc0\x81($\xddfX\xc98\x94\'^\xdc\x95\x11\x84\n\xd0p\xdaQI\xcc@\x9e\x7f\'\\/\xf8/\xda\x9aC\x06\xcb\x01Zc\x8f\x9c\xfb\xb7\xa8\x1d\xb8\x01b&\xc1\xbd}\xb0&\xdc\xb8m\xff\x85\x96\xecM\xf5\xf9\xd7\xb6\x9fV\xc0\x04`\n\x8c\xa0\x1e\xb8\xb6\xbef\xcf\xb7\x10s\xb4\xc6\n\x83\x13\xbeq\xcd6q\x0e\xeb\x01\xf9\xa3\xd6\x07\xf5\xa0\xe4\xe5\xdc\xdc\x89C\xf3\xb3\x0678\xe8\x00\xc4\xca\x81p\xbc]n\xf2\xfd\xb5a\xe4D\x98Wl\xad\'\x1a\xd2\x8e\x88\xdb\xfd\r,\x18\xd0\x19\xa0\xc5ib\n4Z6\x0c\x16\x03J\xd2\xec\x00\xb8\xffr\xc5r:t\xcdZ\x06\xc4\x0f\xb1\xc3\xd6\xa9\x05\xc0\xd1\xe6\xb5\xaf\xdb\x81\x83\xda\x00)\xc1\x17\xd6\x89o%AV\xb8^\xd0?)\xce+\xe1\xd2\xbd\x1e\xf8F\xc9\x80\xc3z\x04\xe0)\x80\x8a\x0b\xde\xc6\xde_M\x11\x0f\x1b\x01c\x93\x02\x19q\xc0\x9b\xecc\xceT\xd8w\x0ed\x07\xe0r\xac8C[D\xb3N\xe0\xa0\x1d${\xf4\x9bV.A\x90h\xa5\xb0\x88\x8e\xda\xf5"Ru#\xec"\x01\xf3\xb0a\x13\x88|\xa0~v\xa0\x1b\xed\r\x9eK\x1e\x13\x83\x1ab\xfa\x13\x1b\xec@\xb7\xa0}\x94D\x04|\x1f\xfa\x08 \xa4|\xa0[\xdd\\|\xc4\xceafWT\x03\xce/\xa7\xe2\x88`\xb3\xfc\xe5\x19\x0c\xe2\x0e}\xc4\x05\xfbw\xa0[\xf8\xe9\x02\x80%\xec\xcc\xa4i\x05&\x1f\xe8\x16\x92\x80\xd2\x01\xd6\xea\xd4\x96\xe0\xae\x1f\xe8f[C\xbf\x07\x08x\xab\x05ar\xfaI\xc0[\xdbXL)>I\xc1F\xf3pU|\xc0\x1bos\xd6$\x19\xbb\x14\xd0\xb2N\xa5\x9d\xf0\xc6I|\xf5-\xa0\x1fpQ0\xb2\x19S5Y\xe8m\xfa7\xbcIV\xe3\xa1J\xa0\x06@\t\x81\xc5v\xc0[\xdf.*I\xbdC&\xb2\x13\xf7\xb5a\x01oc\x83{\x8b\xc1\xfc\xab\x80#\x00:\x11@\xc6\x01o\xba\xe3\x01%%\x0c6\xc4 "\x90\xe4v\x13\xb8Hd\x84\xda11\xdf.\x88/;\xf0M\xd3\xd7YR\x7f6\x80pR\xf5\xc6\xb7\xb9\x80\x96\xa8\xc2\xa3\xa3\x8c+\xd5\x13\xe0ts\x9b\xbf\xaf9L~\x9e\xdb\xdc\xc3%\xe7\xb42\x82\xba\xc12o\x80\x9b[\xf7*V\xf8\x07W\x08V\x08KX\xf1\xef\x03\x02\x82w\xcf\xe1m\xd91\x84(#\xac\xf3\x02\xb8\x1ct(h\xd9\\\x07\xcc\xbd0\xc3\x03?\xa2t\x0f\x1b\x9e\x13c\x1d\x02w\x1e\xa4KZP2\x0e\xdef\x19\xc4\x02d\x1e\x01\r+\xb9hu\xe0\xf7\x12as\xc9\xec\x9c\xc5O\xf1\xc4\x96\xba\tYM\x84\x1d\xf2\xddY\xe4\xa8\x9f\xa4s.\xbb\xa7\xf3\xc6\x81\x0c(-D\x139\xe0\xc2\xf6`N~\x8ayq0\x10\xf0\x91\xda\xa1$2&\x90[$\xba\x10&\xb9\xbb1\xc9\xe1W}[FK[\xd7\xcfhZ7\xaf\xd19\xe73\xa6A\x14v\x1c\x08\xa4\xc8\x17k\xe3\xcd\xea\xbf\x0e\xbe\xf4U\x18\xdc\x12c\x86\x18\x06\t\x00(<BS\xdbK\xd5d\xf4\r\xf6\x8e#v\x14\xba\xe9UX\xfc)u\xc3\x97-\xd6\x18\xa1\xe0c\xa5\xed\xdcT\xca\xb6\xbb *\xc2!o\x191g\x9d\x01\xa0A\x84\x02t\x10\x14\xed\xf6\x858\xd1\xf6\xaa\xf5\xdfG\x18\xd4\xcd\xd1\xa6\x9c\xda1\xd3\xf1\x00n@\xfaE\xf2\xc8\x99f\xd5\xdeM\xb5h\x91\xd3\xfa"\x91\xc6\xb1V\x17\xe3\xd4\x0c\xee\xdboFP\x92\x14/\xdf\x94FI\x07J{C%b\x045\x9cQ\x1bm&\xbcrn\xa5$.\xbe\x04\x8f3:Op\x94\x0fJ\x10\xb9\x89\xf2B^3\xf8\xac\xdd\x82\'\xdc\xb6\x07\xe4\x01\x10\x0cZj\x00\xfe\xb9\xde\x94\xa0\x84\xcfF\xea\x084\n\x9b\xa1\xad\x1e\x8c \xeb\xca\x12\x13kWqN\xc0r3\x82v\x1d\x1f\x04.\x19\x0fq\xb1x\xba\xb8&\x01\xcf\x81\xff\x15a\xbe\x19\xd5V\x0fk\xe4\xfd}\x92\xa2a\xc5ac\x8fm\x0c=\xf0 (\x1dg\xfe\x07\x99\r\xe6\xe5\xd9\x84\x03Jy[\xd1A\xd3\n"-\xb0\xc6B\xee\x97\x14\x94\xdb\xc6\xbc\x95%g| \x80L\x82>\xb43\x97B)\xf1\x8d0\xf8(4\x92z\xc0\x12\xa7ts\t\x9e\xed\x9c\xb5y\xae^\x0f\x14\x89\xddx\x99\x17\x06W\x01\xcf\x18\xbd\x973\xdc\xf3\xf6)M\x87\xf2\'5c\xe9D\xde\x94,\x08+\xa4\xd2\x95\xad);OCi/\x14\xeaO\xa9@\xef\x9fb\xa6n*51c\x89[\xc3\xb4\xb5\x98\xd9\x81\xbec;_K\xa9\x0c\x85\xe8i\xce\xe7\x02\xf6\xf4\xbf_\xb0\xa4<}\xe4\x8dV4.\x15\x84\x1c\xf0RO)\xc3)K\xaf\xc1\xb1\xc1\x1ep\xd4\x0cq\x1bI\xe1\x9c\xca\xae\xfb\xf8&\xa7"\xcfZ\xba\x07vq\xe0\n\x93\x8bT6\'\x99T\xbeT+ho\xeet\xbb\xeaI\x17\x10f \xe1\x97\xa4{\xd9\xf3\xd1\xb7\xd4\xbf$\xcd\x9eWz\xe2\xba\xa4\xacn\xd9?|A\xf5\x99\xd4\xba\xe2D\xdd\x07@k\xbe\xdfw^\xb7\xef\xac\xc9K\x14z1h\x85\xce|\xc0z\xf9L\xa9\xf9\xf1#\xe8\x0b{b\xf9\x86\xf5\xbe\xb7R\x03$\x9b\xe7c\xa1D\x86\x1d\xc0\xae\x91*\x8e\xefr\xc4)\x02\x81Sx\xe8\xadN3<\xad\xb2Ui\x1d\\\xc8\x8c\xf4\x0en\x92*\x19\xd3\xbc\xa1\xbdkuaX\xf9\xced\xc9\x0e\xeb\xe3M\xf4@C\xaa\x0c\x1d\xf4%i>6\x15\x99\x86\x00\x9e\xa7\xcdK\x02\x01\xa9%\xf1\xef\x1e\xfc~\x95\xda\x04\x94\xb3xj\xf8@\xf6H\xca\xcee\xacl$\xe4#@\x04f^\x0f\xab\x19\x9b\x84\x8c\xd8g\xb0\xa6\x82X\x84\xbd\x1e\xa2\xdf3Y\x92\xab\x02\x80\xbc^\x00\r\xc3n\xfb\x89\x02\x8c\x04;%mE\xbb\x97(\xee\x94z\xdf\xb2\xe1\xcddY\xf5x\xe4\xc5\xab/$y\xec_\x17}B\x90\xeb\xd0\x0c\x85o\xa9G\x9b}/\xe5\xd2~\x15|\x1a\xe2\x9cc\xea\x9d~\xa7TzYT\xbdi\x01\xa8\xf7\x95k\xd5\xbf\x98\xfe-\xf5\xfa\x06rM\x19\xd6Z\xbd\xccW\xdb\xed\xe9c\x7fR\xf8\x06\xa86\x14\x10K\x89\x1ca>\x86\xe3\'\x03\xf8\x1b\x8c\x17\x91M\xfaG\x1a\xabmG\x92=3\x16\x80#6\x8f\xde\x07\xf4[\x82\xa6X0"<\xc3\xce\x8a\x98|)\x16\xb7\xbda\xf6\xe6\x8f=\xb8I=\xb0?\xea\tsr[^\x87\x13F8\xa6\xf1\x91\xc8j\x9b\xa9\x07\xa0\xbb\xd8\x84\xbc\x82\xea=\xd1\xdf6\x99\xd5PA\x0cn\x82\xf5\x0e\x06\x87\xd0;So\xfb\xdcB\x00P\'\xff>\xb0\xe5`\xad\x92J\xaf%\xd7\xce1\x9a\x18\x14\tfiz\x80\xff\x81\xfc{p\x85\xf5\x828\xe3C\xfb\x9d\xe3\x0c\xd9[Bo\x96\xc6V\xdd\xd2\xfam\x1c\xbc=\x83vX\x01w&\x80\x90\xa7\x9c\xee\xca\x18_k\xbe\xc3\x00\xa5?\xe7=\x1ftqsrmz{\xf11\xfc\x83\xf9\xfd\xfeR\xac\xb6\xad\x84\xa6\x08\x03\xdd\x02\xec \xc4\x96v(\xe3~*\x9bi\xc7<\x9c<\x8c\xb5\x07\x942t\xf7\xd8\x92\xd7\xd1\xeeRn\xfd\x98\xb9\xdc\xd6\x1b\xbc\xac\xdeE\xcb\xccM9Wb\x19\xde\x00\xd4\xf0\xc6\x0c\x95/\x15N:\xc3\xaf>*\xbf9\xea}\xebp\xb8\x168\xa4\xb0\xab\x85U\xc7\xa6\x0f\xc5\xc6\x01\x8e\xe5\x97\x8b\xfc\xce\xa0\x90b\x1cA\x99S\x84\x8bl\x7f\xd9\xfd!T\xc0\x86\xb8\x8d\xa3M%tK?\xf2w\xac\x9er\xac]\x8e\x08\x17s\xd6\x9c\xe6V\xf0\x87\xce\xd4\xc4n\x98\x1e\x89\xee\xadex\x81\xdc\x08\xf4P\x8f\xf3\x95\x9c^\x0e"\x81\x9f\x0f\x1aGE\xe8\x0b\xf2\x1cy|\xaf\xd4\xc3/\x8b\xa8V\xfe\x12;k&eN\xed+\x1c\xc8\xeb*\xfd0\x1cN\xc2,\x12;\x00\xca\x82O\xdc\xaa\xef:\xc4r\xf6Ox\xc0jXH\xb9t\x80\xed\x0f<\xda\x81\xd8\xd5\xaf\xe0+\xcf6\x95\x9e\xa8}{\xcb\xefZ\xec\xc9P\xf6\x03-[b\x9d\x12\xda\xc5\x85\xbf(\xac\xf5\xcc\x8a\xc5\xd7e\tU{\xf7R\x9e\'\x99n\x07\xd0\xe4\xffKO\nP\x07\xa6J\x85o&Q\x92lX\xe4\xd7)U\xf5D~="-%\xbe\x1e\xb5X\x83\xb8 \x18\x13\xb7v\x84\xb0\xff\xe6\x03\xf9d\x99\x9cK\x15\x9ep\x1f\xb0e\x1f\x7f\x175k\xf2\xd9\x11\x1a\x03^(\x95\xf5V\xaa\xb47o\xec\xc1\x10\r^|\x80Q\xd3]\x97\x8e\xde\x90\xb7\xdd\xc9;>H\xc9N\x8e9\xf6\x8a[\xae\xcc\x95&\x05t\xcdZ\xa4\x1e\xe9\x04oJgB\xc5\t|u\xc3;\x10\xb6\xa6D\x13\xefmn\\\xea`\x04G\xba\x1b\xc5j\x94w\xa3\x0f\x07\x9c\x02\xdc\xd8\xac\x9e\x1eH\xc9\x9a#\xd96\x9akTH\xf7\xd9\xc1t5\xd6D\x8b\xc8\xa4\xbaP\x93\xcc\xc0\x02\xe5\xfe\x05\xbcb\xfe\x95 \x94\xdb.\x83h\xc7RA"T\xc1\x88\xc9\xce\x16\x810\xca\x7f\x00\xe6(\xee\xe86\xe4\xb0b\x90a\x1du\xc8\xdb\xe7\xd4>\xc2\xcc8bH9w5\x1aK\xfe\x10\xe3\xe8l\xe8\xb0\x94)v\xd9-\x90\xb2V\xcev\x9e,\xf7R\xfc\xfa\xfd\xa5\xdd\xado\x8e\xbf\x1a/\xf0-,\x08b]\xbf\xe83\xdb\x1ct,\x08\xc6!5\x01A\xea\xdf\xcft\xe4\x166PD\xaa\x04l\x95\xbb\xd3 \x12\xd5\xd1z\xd4a\x00\xc5G\xab~\xa9\x1e\xd2\xce\xde,\x99\x08\xad] \xffJ\x14\x94\xca)\xfd5\t#\xec\xec0|\xa5\xea\xa9\xcf\xda\xceR\x8c8\x8b)@\tQ\xa65\xbd\xf5Y\xcb\x04w\xd5\xe2\xbc\xf0*8?\xbdc\xccH9\xd0\xb1#\x01\xc6Y!\xbbIe(\x82\x1e\x14T\xa4\x83\xccX\x9b\xfdn\x92\xe2\xd1}\x80\xf4\xcb\xc0\xc7\xbd\xc5\xa7\xd2\x07\xa9\xa0$\xfcvs\x9cV\x98P/\xed\x94g\xb2\x93X9i\x02E\xe9\xf1\x85\xbb~k4\xb8\xfa\x92\xba\xd7F\x10\xf7\x17I\xb8\xe2~\x94\xa1\x16\xe8@\xaa\x02\x7fe\xe5ds\xa7A\x00\xe5\x98\xd8\xde\xa0\x12\x81\x0b\xc3\x13"\xfd\x8e\x1a\xe5\xc3B\xf1e\x98\xda\xc6\xb0\xfa\xa5\x8d\xa0\x9d\xc9P_\xc4x\xea\xc5\xfa%\xd3\xd2\xa3\xce\xb7\xd6\x0c^#\xd6V\n\xa9%-#;o\xfa\x8f\xfa\xab\xa6\xaad\xe0\x99)T\xa0xa\x80\x0f\xfd\xd5?w\xc3+\xf8]\x1bHS\xbf\xf4\x97\xa4\xbci\xca\xbd\x00\xb0-\x96Q\xce2m\xa8\xdbH\x93{\xabH\xa1\x18\xac\xc9\xea\xc6\x0e\x05\xfc\x875_\x89\xe4\x1a\xf6\xec\xd97\xf5\xfe[o.\xbb\xf5W\xe8\xe6\x16\xc6\x8f\x88\x0b\xd6[\x8cZd\xdf2\xdf\xa4\x8ds\xbcc\xaek\x15\x06 \x1c\xc1\x8eRoh\xb8w3\xa8F\xcc\xdd"\x80\xf1G- >\xb1vl\xb2\xcb\xf3v\x16\xdar?k}a\x14\xba\x1c\xac\xb7\\\xc7\xd2R\x92$fv\x13\xf5PG\x9f0\x1a\xf9\xa2\xa8qW\x11\x81s\x97N\xfdK\xfa\xe5\xa2\x90\x00;\xc4\xb2\xee\xc4\xf06\xff\x1c\xf6V\xfe\x142wL.t+BK\xfd \xab8(\xde7\xd2V\x11\x96\xcf\xbc\x8e\x9cL\xdd[L\xa82\x84n\xbb\xf5\xe0}\x80o\x84l\xa9\xf7"\xba\x96u7\xa2x\xb7\x9f\'\x92\xbf\xf8)\x9d\xc7\x06\x03\x1685\xac\x99\xeeH\xc2\xa97H\xd7\x19\xd3\xd3|\xa1\xb5}/\x8b\xd5\xa0J.\x8a\xef\xdch\xf9\xd8\x83\x88~\x05\xd1\t(\xe4\x02 \x1abJ\x02\x8c\x91\x93l\xde,\xe9\x1d\xdf^o\xe1\xbb\xfa\xa5\x91S\x89\xc1\x90\xba\x90\xcfb\xa7\xce\x8d\xaa\xdfJjL\xbc\x98\xc9WP\xef\x93\xcd[\xca\x19\xe9\x1c\xdc~\x15|\x1e\xc3\xff\xcb\xe8\xc7\xdeEF\xd0"\x96\xf8`L\xec\xe4\xf8\xec\xb3\x96\xbd\x1b#%\xaf\xf0q\xee\xd3p\x91~K\xc8H\xbf\xe8F\xda\xd1\x07\x9c\xbf\xe3L\xfa\x11\x84c\xcbrv\xb2y+\xfb\x90JQ\xc3\x0f\xea\xd6\xf7f\xe8N\xf5a\xad\xb0\x0c\xef\'\xe8w\xf9+\x9f\xca\x9c\x19$\x00\x885\\\x0e}i\x88\x89\x889\x07\x0fOix\x81\xdb>\xca_\xb2\xf9\xc8{&\xd5 l\xbcE\xe8\xb4\xba\xd8\x8al\x19\r\x80\xe8\x12YTnU\xa8\xe9\x03\xe7\x99x\xbf\x01l\xb4\xb1\x9dQ8Lc$\xb9\x02\xd3,\xa0\xe7\xde\xb2\x7f\x18\xa8m\x87\xea\xa9\xe5\x0e\x04\rA\xa7\x8f\xfeQ\x8f\x8e\xe2\xf0\xdbF;d<\x05\xa2\xc1G\x92\xf4\x8b\xe5\xdfI\xd2\xf7\xe3\x93A@O\x11\xc2nYiDN\x8a-\xf6K\xdev\xaa\xceC=\x17r\x16\xc8\xde\xd3M\xa5\x1e2\xcfU \xae\xcb)2{J*\xf7\xd8\\Du\x9cBi\xad\xdfy\xd205\t\xef0/\xeb\xd9h\xa4w\xe7\xc5H\xad\xd2c\x05up\x8b\xe1\x1b\xdcn\xe1h\xc9\x9d\xd62\x10\xa6\xbd\x97\xeaU\xd1z.c\xc4\xb1}[\xb3\xa4\xdd\xb0=\xb3\xee\x9eNpv\xc8\xee~\xf6\xc3\xc4\xd6\x8d\xd4\xe0\xda\xbcz\xcb\xb0!\x88\xdd\xdf7\x86\xdbN\x86\x94\xd4\xf0\xdb\xe1!\x9d\xe8\xbc\x18\xc5\xa1YR\xc9W\x9b\xb7\x82\xb7\x82\xd0w\xc0Et\x06\x95\xe3*I\x11,\x19\xcc\xa5\xdf%_\xdb\xa17j\xad\n\xe0\x94\xe7\xc2S\xbb3N-D\xe1\x8ey\xf0R\x82\x1ah\xabh\x94\x83z\xce@\xeeN\xdb\x82\xe8\xef\xb9\r\xbb\xd5\x80\xa6\x1eO\xda\x17\xa3\xc8\x9b\xb6\xad\xd9]\xe8\xe2\xacxv\x9f\x81\xc0\xfaw\xeb!\'\x05S\xd3n\xac\x99A\x14\x04J\xaaT\xb9\x1b\xcd\xe2\xebJ\xf8\n6\x02\xba\xbbV\xfa\xc88\xbd\x17\xcab\xeb\x10u\xba\xa7n\xa3\x84\xa6)\xb0Fra\x15rU\x1c\xb7@\xb6\xe8\xb6;\x8e\x8aQ\x08z\x85\xa2\x83\xf8+\xf5\xa3i3\x92\xe2\xfa\xf6\x0e4/2\xc1D\xec\x80\xe5\x9a\x8e;z#@C\xbcT\xde\xd7\x07\x86\x89\xd6\xd8\x87t\x01\xc2@\xb4\xd8\x01A\xf8\x00\x83\x91\xba\x0c\xca\xdb\xd2|| \'C\xa2d\xa5\xbb]\xa5\x81\x0b\xe0\x1f\xe2#\xaeF\xbfr\xcb\x9d\xa6\xde\xbc\n\xc2\xaczwK\xdaGOc\xc5rk\x83\xea\xa23\xcdA\xa9\xf1#\xba.\xa0\x06\x10x\xa0\xff.\xfe\x12W\xebr\xbf\x8a\xb7\x0e\x00\n<32\xdd[\xbf\xe5\xb6#\xf9^\xdc\xa9\xffaH\xa4#\xdaG\xfa8z\xd0\xb8\x14\x03\xd5,\x95\xea\xdd7W\xd3\xbd\xb0\xd5\x0b\xc1\xde\xdcJ~\x87\xe7f\xe8\xb4g\x8e3x(\xc2#\xd7?\nWQOy\xdb\xcat\xe8@L\x1f<\xbed\x84\xa3\xdbw\xf7\xa0A\xa2\x82G\x96\xd2o\x05\xfe\xa6\xdd\xa3ubxn\xae5\xfa\xe0\xd1\xb9\xd1v\xafB=\xb9j\xc5\xf4Nb\xc5f\xbc9B\x10\x9b\x06\xa5\xa5\xe3\x03\x7fzZ2\xef^Xo\x18\x02\x95m\xb7\xa6\x8eK\x12\xaf\xa9\x03x\xfc\xee\x1d\xf1G\x9ftMt}\xd9\r\xb0\xca\xc0\xc9\xca\x95\xce\xa2\xb8\x86\x92\x96\xd1\x06\xb0\x18\xa1lH\xbb-\xd2\xe2\xf4\xa2\xd1\xd5{\xc6+\x84\x13"\xea\xef/:\x80\xcfV\x12\xee\x9e2\xf5\x84\xf3\xb8-\xa9\xa7\x9c\xf7\x9a\xfb\xb9Ue\xe35\x8e\xab\xab&\xcf\x0c\xf06\xef\xb7\xa7\xb8N\x9a+s5\xda\xe9V{\x00\xfeaFd\xb0[\xaf\x87\xe5\xbd\xf7$\x9cSc\xf7\x94\xce\x9c\xa9\xa5\x80\x93n\x82A\xcd\xc2Q\xba\xddQ\xa1\xa4\xbe\xff\xb5\xd1\xc0\x9e\x06\xd7jWT\xe0\xd4P\x12\xf2\xad\xc3\xe2\x14\xf2I\xc7G\x0fm\xde\xe9U\xe1(^\xd4\xf4@\xd2\xbe\xdcj\xa5\xbd\xd9\xbb\xabQ\xaa\x92[i\xbf\x85a\x8b\xecx\xb4\x9f\x9d\x8d\xae\xb9\x05;*\xc7\xb1f\xfc\xf8\x06\xef\x06v\x7f\x14\x00\x02\x86\xde\xe2\xb1pS/\xdc\x15\xbd;\x1dG\xb2\xfe\xbdfSp9xb\xbd%"\'\xa3\x9b\xb9aD\xdf*\xd0\xc9\xab\xcb\xaf\xa5\x03\xa4t\xda\xfbZ\xab\x1b\x06#8P\xa4\xf5\xf9\xbc"Z\xe2\x13\xf9\x97\xd7\xb0aKDS"\xb6\xb3\xe3\x96\xbe\xb4\xfa\xfc\xa9^:\xaf\xbax\xd6\xd0\xfb}[;\x01\x91S\xce\xfd\xa3\x16\xca\x1fy\xec\xf2\x87\xd6\xa1\x9c\x17o\xd7\x0f\xcf\xc1 K\x03\nA\xb3\x80\xc7\xef\xcf\x0eh\xf9S\x14\xdb\x8eV\xfd(\x015\xc7\xcbV\xae\x86d\xdd\xfeHY\x81\x89w\xc3b^\xbdEq\x0c\xb47\xe3\x06rIe\xab\x99\x9c\xcc\xcf\xeac\xb5\x0e\xb9^\xf4\x1e\xa6\xda\xee\x9ePI\xaax\xe1;V\x8b\x7f\xb9Z\xbb[\x87l\x8f\xb7\xb7\xd2\xd6\xfc\xf2\x90\xf1u\xdf\xd6r\xe8\xde\xe4\x9c .\xbf\xf5\x84\xe6\xaf{S\x8a\x02\xdb\x1e\xec)\xb7C\x15\xcbN\x1a\xbd\xed\x11\xf7\xcd\xdb\x10(\x162\x02N`\x0f\xda\xf4U\x87\xe7\xa4\xc0t\xb7\x94\xc4I\xf8\xd5|\xf0kW\x96\xb7\x16~\x7fr\xa4k\xfc\xe6\x02\xd4`\xbf\xee\xe0\xf6x\xa3 \xc9[/\x86\x17\xef\x8ahr\'\xfb,]\x95^\xb7Dq`V\x07\xa8\xa5\x1eZ8$\xe0\xbaU\xbaT6,\xcd\x9c!\xf1m\xe9\x9a\xf2\xf9cG\x8bn\xe4\xe9\xd2/Z\xf80\xf8[\x0b_f\xbet\x14\xbe~x#}9sa\xd1\xaccI87\'\x05\x056\x13\xd5K>\x7f\xb8\x86T\xa4_O\xdb\xfdk3t^o\x8cK\x8e\xcb\x81\xc8+\xe6\x0c\x0b\xd6;g\x1d\xf7\xb2\xe2\x86\xdaP\xcfv:\xd8}\xdcB\xa0\x94\xb0\xa2\xb8\xd3Cj\xf4^\xd3\xe13G\xd3s\xd7\xdb\x00\xe71\x02\x01\xab\xb7&\xe9\x1f\x97!>\xd7,\x89\x88\xea\x9e_V\xb7\xbe\xdf!\xad\xd2#\xbc^\x87\xa2\xa9\x12\x94\xbb0\x7f\x1f\xfb\xf1\xf6\xa5\xe56\xc7*~\x81\n\x8a\xce\x0e\\\x8bzTO\x88\x82\xbdC\x04t\xde\xca\xf7\xf5\xb7(\xc1\xa42\x1aT6y1\x88\xef;\x06\x81@\xaf\xebu\x10\x19\xe8\xa8\x95)\xc9\x85pN\xedlkf\xa8\xbf\xe1]\x1d\xfc!\x1ez\x82+\xdet\x8a\\\x82\xb6\xf3r\xafD\xcd/\xe7kF\xf7wJ\xc0\n\xce\x0b\x17u[\xc6\x91\xcdl&\x05T\xd4fo\x82\x9cw\x01/\xf1gg\x02\x8f\x13\xa9\xcb\xcbX"\xdb\x8b\xcf\xfe\xbe\xc38\xdf\x02\x08u\x9b\xd3\x99\x0c\x9d/\x9d\xb4i\xbd[\xd29e\xaf\xa2\xaf\r\x11\x0c\xeb\xe6\x03\x86FjS\x8d$\x02(\x8f\xfa\xdd\xfe7\x86fB\xac[\xd2\xbd\t\xd8\x82%D\x03t;\xbbVzB!0Vo6\xe8\x8dO\x02XSy\xee\xed\x9e|\xb2\xe2^\xd1\x90\xfbY\x13IW\xa3V\x13\xbd_&\xc3\xaaE\xc6]\xeaz\xf3\xba\x91\x19<\xcf/\xa7\xd0,e\xba\xd7eU/d@U5\x1dw\xbbAOAG\xc2M\x1e-\xa8g\xc4\xcb5\xe1@\xfb\x01\'\xe9Xt\xb3\x0f\xf1^S\x1dm\x81\xa8\xf7|B\x07\x9b\xda\x97\xae\x8eXs\xdf]+\x02K\x82\xa4y\xabWY\x00\x85\xc8[\x0c\x00H\x80\xed+<9\xdd\xb5\x1f%7\xaa\xf2\xaf\xe1\xf2\x84\x9f;e_\xe8B4L\xf5}\xc5\xe2\xb9\xe6\xdf\xdbiv=\xb1\xbbx#\xa1\x01\x06\xf0\x9f\xae\x1a\x13\x9d\xecYr\x17\xcf\xf3@N\xeb\xb5\xf8\xeb\x0e\xb7\x90\xb6t\xf7f\xc9]\x1a\nM\n\xff\xfeh\x90\xab\xe9]\x1f[\xa6\xd4\x81\x8a\x90\x12\xf5\xa3\x81\xcfv\xa9\x8b_!\xed\xf8\xecM\xfa|x7\xa5.\x97\x11\xe9(o\xed\xd0\xceK\xc2\xd2\xc9Mu\x7f`\xdf\x97\x8b\xc8S\xdd\xd2\xea\x97\xbaw=\x0b\x7f`\xfc:\xd8\x0b\xea\xe3\xcb\x8d\x85x\x08h]C\xf0\xdb\x83\xe69\xd0v;\x16E\xac\x8a\xdc\xceG\xdf5\x9dO\x00q\xbaa\xe7T\x99\xfcZ%\xf5/%\xb7\xc3\xea\xc0\xe1\xcc_\x1c\xf1B\xe1\xc9_zz\x14&\xdee\x11Dm\xbf\n[\xed~\xf2\xa3m\xd4\x88f\xa2\xeb\xfb>UX\xae(\xe2\xe7\xbb\xac"\x1at\xab\xb0\xa8=\xf0\xfb>\x08,\xdf\xc0\x13\xb5~i\x80\xa2\xf3A\x0c\xf2\xdc\r\xa6\xa7\xd6\xee\xb27\xef\xf4\\4\x01`\xcb\x00\x03\xdd\x9f\xf5\xb9\xabz\x96h\xe5\xce\x13x:\x11\x98\xbb\x9f\xf1\xc8\xf4d|$6\x9f\x97D\xcc\x93\xac\xf5\xae\xeaq\xda\xbf\xd5Z\x0e\x9b\xf3\xfd[/*\xe4@\xdfR\x17\xedN\x12\x0e\x852\xdd!\x96?\x80\xb4\xe7\x0c9\x10\x0c\x02\xd9\xdf\xae\xf9\x12\xe8\xe3Z\xf5n\xae\x11\xbf=\xeb\xcf\x84\xdc\x8di=9\xad\xee\xcb\xc7\xde\x97&\xdd\xf4\xbe\xe1\xf76\x0c\x84_U\x08\r\xcf}}*\x98\x91\xca\x11\xab\xfd\xd9;\x9d\x8a\xfa#_\xb7\x82\xd1d\x1d\xeb\xda^\x15\x00\x92y\xee\xe4\xae\xeb\xd5\x94\xdb[\xd7{\x88]\xcb\xb5\xd5\x85-g\xe7]\xf4r\xd0\x16\xf4e\x8c\x01\xa58\xbe\\t\x8fJG\xdfOO\x89\xff\xaf\x95zs\x08\x8b\x91\xe1\xde\xde\x8c\xa0\xfe\x8a\x07\xdd}_\x9a\xaf\x94-#\xad\x9eu\xb2\x11\xad\xcd\x9cZ\x8d,z\x17\xf6ubV\xc8\xca>\xdaA"B\xc8P\xcc\x0c\x12\xd1Ke\xb8\xd6k\x1b\xf9\xf6\x03\xe7\x92\xa8\x07Y\xc0\xad\xe77\xec\xe3\xa2\xfbm\xfc\xe5\xf3\xaa\xd3H\xaf\xb8\xad>\xa0\xa2\xc5[N\xea\xd9\x7f3\x92\xb9\xbfw\x88\x87\'\xc6\xf0]\xa3\xdf\xb9\x92\xb1\xf7+\x9eI(]\xbc\xc4:\xa2\xce\x9bw@\xf2\x0e<\xc6\x030\x84m\x96vK\x19K\xdb\xb5\xe2{\xf34\xa19\xc2\xdc7\xdf\xe2Z\xce\xdb8\xedj\xdd\xda\xee\x0e\xcf\xd9\xf9\xe8J\xd5\x18\xdc\x9b\x15o\xcd\xba\xde\x9d\xd3\x84\xf5\x7fY\xb3\x9c\x89\xd3\x9e\xca\xb1\xf0~\xf5\xc2t@\xc5U\xd5{w\x99\xe7\xe32\xe2\xd7O_%\xa3\xa7l\x1e\xe9R\xce\x9f24#\xdd\x17\x7f_\xb9\xf9C\x86\xa6&u\x99\x98~}\xc4\xe8{\xfd\x81N\xd93\x8egM\xfe\x98\xa2\x91\xa4O\xf4\xaf)\x9a\x9a\xa0\xf3\xbd\xb6\xebD\x1f\xe1\xa17\xfe\xf2N\xd1{\xb9\xea\xaf)\x1aI\x8f\xa7\xc5\xd5\x81?\xa5h\xfa\xc6\xd9\x91\xb8\x80+h\xf5\xde\x9e\xd1\xbf\xdc\xee\x92|)\xe1N\xd1hR\xa0\xb2\xa9\xe4\x9fR4\x96\xfaX\xd2\x1bB\xdfS45]S\x8av\x85Z\xadKG\x04\xb9\xc8\x93\xa6\xd7\xd7\x82-\xff)E3\xd2\xee\x86\xe8B\xac1?\xb5\x1e\xef@\xe4&\xdd\xb7)\xf1\x9fR4\x92z!\xfa\xdfR4_,\xfe{\xbbB<\xbeb\x7fM\xd1\xe8i\x06\x0b\x02\xa5\xb4\xe2\xc0&w\xb7B\x9cm\x1c\xec\x9f24\x9a\x92Tq\x10\x7f\xc9\xd0\xe8g\x8e\xf3\x0f\x19\x1aI|\xf7=\xb5\xee\x97\xde\xc1j\x06\xdd\xb7\xba$I\xb3\xbfgh\xa2\x07\xbc\xffu\xcd\x92\xbcM\xa2\x02\xba\xdb\xdc=v!x\xb4/\xb8f\xe7E\x9do\x19\x9a\xcc\xbd\xe3\xe1\x9a\x95\x9a\xf7\x82\x9eU\xe8-\xb9[\xe8{z\xa1rU\xde\xcc\xd9&\x90\x93\xce+/\x94._\xd9\xfb\xa2&\x00\xa5\xfb\xf5w>\xa4V&\xde\xef\xcc\xbe\n\xf3GU\xcec\xe99\x9f\x1e\x01\xc4_\xe4\xf0\xd7&\x0b\xff\xb1f\xf9v\x156o\xc9\x84\xd6j\xe3VO\x9c\xde2[\xa5\xd3\xa7\x12\x03X#:\xac?\x7f`\xf4\x14\x80=\xfa\xcdw\xa9\xfa\xd1UX\xafKn\x8f\x06\x18\xcc\xddk\xe9wO\x7f\x8bN\xbd(\xf7>\r2\xae\\\xe8n\x8b\xa6k\x9f\xff\\=\xa2\xd4\x07\xf1\xbe0\xe5{\xd1\x04\xdc\x86\xee\xf7\xc4rs\xca:A\xbf\x87\xd1\x9fg\x98\xbe\x94C\xfb\xf9\x10\xaf?/\x85\xad\xb3Z\xf8\xfb\xfd\x91\xb7D\xcd\xbf\xe4Kf-:\x15\xa6\x87\x143\x01S\xa8\\)\x8a\xa0\xed\xa3:\x17\x97\x08\xc9\x1f<\x1a\xfe\xc4\xe9y=\xb1\xa6W\xdaJ\x10cX=\x048\xe8\xc2\xc71\xdfEP\xfe\xe5\x1dH\xbe\x05\xad\xe9\xbde9\xbf\xb0X\xc50\xef\xd9\x1fZ\xfa\xdd\xb3\x91\xd7LGC\xa8$K\xa0|}\xcd;1\xbd\xe4>v\x07d\x8eH\xb9\x94X\xe2\x1a\xc4\xef\xff\x07x\xc4\xa6L'))

class Dummy(object):
    pass
myglobals = Dummy()  # Container for shared arrays

# Log2 lookup table
LOG2LUT = np.log2(np.arange(1200000))
LOG2LUT[0] = 0.


def log0(x):
    return LOG2LUT.take(x)


class EntropyWorker(mp.Process):
    def __init__(self, rank, size, queue, max_templates, max_time):
        """
        Process worker that computes entropies
        """
        super(EntropyWorker, self).__init__()
        self.rank = rank
        self.size = size
        self.q = queue
        self.max_templates = max_templates
        self.max_time = max_time

    def run(self):
        """
        Compute as many entropies as possible in the given time.
        """
        N = myglobals.N
        t0 = time.time()

        # Loop through states that belong to this process
        for k, i in enumerate(myglobals.random_state_id[self.rank::self.size]):

            # Get template
            template = myglobals.allstates[i]

            # Compute matches
            m = (myglobals.solutions == template).sum(axis=-1)

            # Compute color matches
            ctemplate = np.array([(template == c).sum() for c in range(myglobals.Nc)])
            S = abs(myglobals.csol - ctemplate).sum(axis=-1)
            c = myglobals.Np - S/2 - m

            # Combine them for binning
            outcomes = np.bincount(m*myglobals.Np + c)

            # Compute entropy
            entropy = sum(ni * log0(ni) for ni in outcomes) / N - log0(N)

            # Put the result in the queue
            self.q.put((-entropy, i))

            # Exit if some conditions are met.
            if (self.max_templates is not None) and (k>= self.max_templates):
                MMsolver._announce('Max number of templates (%d) reached' % self.max_templates)
                break
            if ((self.max_time is not None) and (time.time() - t0 > self.max_time)):
                MMsolver._announce('Time out after %d templates' % k)
                break
        return


class MMplayer(object):

    def __init__(self, Np, Nc):
        """
        Mastermind player.
        Np: number of pegs
        Nc: number of colours
        """
        self.Nc = Nc
        self.Np = Np

        self.initialize()

    def initialize(self, solution=None):
        """
        Get ready for a new game. Pick a random sequence of colors. 
        """
        self.Nmoves = 0
        self.solved = False
        if solution is None:
            self.solution = np.random.randint(self.Nc, size=(self.Np,))
        else:
            self.solution = np.array(solution)

    def new_move(self, move):
        """
        Receive a new move and return the results.
        """
        move = np.asarray(move)
        self.Nmoves += 1
        m = 0
        for i, j in zip(self.solution, move):
            m += (i == j)

        S = 0
        for c in range(self.Nc):
            ss = sum(self.solution == c)
            sm = sum(move == c)
            S += abs(ss - sm)
        c = self.Np - S/2 - m

        self._announce('Result: %d good, %d wrong place' % (m, c))
        self.solved = (m == self.Np)

        return m, c

    @staticmethod
    def _announce(message):
        print ('Player >>>>> %40s' % message)


class MMsolver(object):

    def __init__(self, Np, Nc, mp=True, max_templates=None, max_time=60, solonly=False):
        """
        Initialize with Np pegs and Nc colours.
        """
        self.Nc = Nc
        self.Np = Np
        self.N = self.Nc**Np

        self.max_templates = max_templates
        self.max_time = max_time

        self.mp = mp

        self.solonly = solonly

        self.results = []

        self.initialize()

    def initialize(self, start=None, compute_entropy=False):
        """
        Prepare everything for a new game.
        """

        if self.Nc*self.Np*self.N > 1e9:
            raise RuntimeError('System too large. Sorry.')

        # Create all states
        allstates = np.array(zip(*np.unravel_index(range(self.N), self.Np*(self.Nc,))))

        # Initialise possible solutions
        self.solutions = allstates.copy()
        self.allstates = allstates
        self.validmask = np.ones((self.N,), dtype=bool)

        self.start = start
        self.compute_entropy = compute_entropy

        # List of moves
        self.moves = []

    def choose_move(self):
        """
        Compute statistics and return a move.
        """

        # Update the color info
        self.csol = np.array([(self.solutions == c).sum(axis=-1) for c in range(self.Nc)]).T

        nsol = len(self.solutions)

        # Special case for the first move.
        if not self.moves:
            if (self.Np == 6) and (self.Nc == 10):
                move, entropy = FIRST_MOVE
            else:
                move, entropy = np.arange(self.Np), np.nan
            self.moves.append(move)
            self._announce('Move : %s (entropy: %f bits)' % (str(move), entropy))
            return move, entropy

        if len(self.moves) == 1:
            if (self.Np == 6) and (self.Nc == 10):
                move, entropy, _ = NEXT_MOVES[self.results[-1]]
            else:
                move = self.Np + np.arange(self.Np)
                entropy = np.nan
                toolarge = 2*self.Np-self.Nc
                if toolarge > 0:
                    move[-toolarge:] = np.random.randint(self.Nc, size=(toolarge,))
            self.moves.append(move)
            self._announce('Move : %s (entropy: %f bits)' % (str(move), entropy))
            return move, entropy

        if len(self.moves) == 2 and (self.Np == 6) and (self.Nc == 10):
            move, entropy = NEXT_MOVES[self.results[-2]][2][self.results[-1]]
            self.moves.append(move)
            self._announce('Move : %s (entropy: %f bits)' % (str(move), entropy))
            return move, entropy

        # Special case for the solution
        if nsol == 1:
            move = self.solutions[0]
            self.moves.append(move)
            entropy = 0.
            self._announce('Solution : %s (entropy: %f bits)' % (str(move), entropy))
            return move, entropy

        # Randomly pick templates from solutions and available states
        random_state_id = np.arange(self.N, dtype=int)
        if (self.solonly is True) or (nsol < self.solonly):
            random_state_id = np.random.permutation(random_state_id[self.validmask])
        else:
            random_state_id = np.concatenate((np.random.permutation(random_state_id[self.validmask]), np.random.permutation(random_state_id[~self.validmask])))

        # Compute maximum possible entropy.
        max_ent = min(log0(len(self.solutions)), log0(self.Nc+1)+log0(self.Nc+2)-1)

        if self.mp:

            # Put all useful arrays in the globals dict before forking
            myglobals.allstates = self.allstates
            myglobals.solutions = self.solutions
            myglobals.csol = self.csol
            myglobals.random_state_id = random_state_id
            myglobals.N = len(self.solutions)
            myglobals.Nc = self.Nc
            myglobals.Np = self.Np

            # create queues
            q = mp.Queue()

            # create processes
            numworkers = 4
            processes = [EntropyWorker(rank=i, size=numworkers, queue=q, max_templates=self.max_templates, max_time=self.max_time) for i in range(numworkers)]

            # Run the processes and join
            for p in processes:
                p.start()

            best_entropy = 0.
            # Get the results from the queue.
            go = True
            while go:
                if not q.empty():
                    r = q.get()
                    if r[0] > best_entropy:
                        # Better candidate
                        best_template = self.allstates[r[1]]
                        best_entropy = r[0]
                        #self._announce('%s (%f bits)' % (str(best_template), best_entropy))
                else:
                    go = False
                    for p in processes:
                        go |= p.is_alive()

            for p in processes:
                p.join()

        else:
            best_entropy = 0.
            t0 = time.time()
            can_exit = False
            prog0 = 0.
            for k, i in enumerate(random_state_id):
                template = self.allstates[i]
                entropy = self.entropy(template)
                prog = np.floor((1000.*k)/self.N)
                if prog != prog0:
                    print '%3.2f %%' % (prog/10.)
                prog0 = prog

                if entropy > best_entropy:
                    # Better candidate
                    best_template = template
                    best_entropy = entropy
                    can_exit = True

                if can_exit:
                    if (self.max_templates is not None) and (k >= self.max_templates):
                        self._announce('Max number of templates (%d) reached' % self.max_templates)
                        break
                    if ((self.max_time is not None) and (time.time() - t0 > self.max_time)):
                        self._announce('Time out after %d templates' % k)
                        break
                    if abs(best_entropy - max_ent) < 1e-4:
                        self._announce('Reached max entropy.')
                        break

        self.moves.append(best_template)
        self._announce('Move : %s (entropy: %f bits)' % (str(best_template), best_entropy))
        return best_template, best_entropy

    def new_result(self, mc, move=None):
        """
        Receive result, reevaluate possible solutions.
        """
        if move is None:
            move = self.moves[-1]
        else:
            self.moves[-1] = move
            self.csol = np.array([(self.solutions == c).sum(axis=-1) for c in range(self.Nc) ]).T
            self._announce('Entropy : %f' % self.entropy(move))

        self.results.append(mc)
        (allm, allc) = self._mc(move)
        selected = (allm*self.Np + allc) == (mc[0]*self.Np + mc[1])
        self.validmask[self.validmask] = selected
        self.solutions = self.solutions[selected].copy()
        self._announce('Just learned %f bits.' % (log0(len(selected))-log0(len(self.solutions))))
        self._announce('%d remaining possible solutions' % len(self.solutions))

    @property
    def solved(self):
        return len(self.solutions) == 1

    @staticmethod
    def _announce(message):
        print 'Solver >>>>> ' + ('%40s' % message)

    def _mc(self, template):

        template = np.asarray(template)
        m = (self.solutions == template).sum(axis=-1)
        ctemplate = np.array([(template == c).sum() for c in range(self.Nc)])
        S = abs(self.csol - ctemplate).sum(axis=-1)
        c = self.Np - S/2 - m

        return m, c

    def entropy(self, template):
        """
        Compute the entropy of the possible solutions conditional to the template.
        """

        N = len(self.solutions)

        (allm, allc) = self._mc(template)
        outcomes = np.bincount(allm*self.Np + allc)

        # Compute entropy
        entropy = (outcomes * log0(outcomes)).sum()/N - log0(N)

        return -entropy


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        Ntries = 1
    else:
        Ntries = int(sys.argv[1])
    Np, Nc = 6, 10
    mmplayer = MMplayer(Np, Nc)
    mmsolver = MMsolver(Np, Nc, max_time=1)

    tlist = []
    nlist = []
    for i in range(Ntries):
        print 'New problem'
        mmplayer.initialize()
        mmsolver.initialize()

        t0 = time.time()
        while not mmsolver.solved:
            move, entropy = mmsolver.choose_move()
            result = mmplayer.new_move(move)
            mmsolver.new_result(result)
        tlist.append(time.time() - t0)
        nlist.append(len(mmsolver.moves))
        print 'Solution is %s (check: %s)' % (str(mmsolver.solutions[0]), str(mmplayer.solution))
        print 'Solved in %d turns and %f seconds' % (nlist[-1], tlist[-1])

    print 'Avg time: %f' % np.mean(tlist)
    print 'Avg number of moves: %f' % np.mean(nlist)
