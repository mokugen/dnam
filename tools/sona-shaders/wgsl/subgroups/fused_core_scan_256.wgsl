struct type_7_ {
    member: array<vec4<f32>>,
}

struct type_15_ {
    member: u32,
    member_1_: u32,
}

struct type_17_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: u32,
    member_5_: f32,
    member_6_: f32,
    member_7_: u32,
}

struct type_18_ {
    member: type_17_,
}

struct type_28_ {
    member: f32,
    member_1_: f32,
}

struct type_30_ {
    member: array<f32, 4>,
}

struct type_31_ {
    member: f32,
    member_1_: f32,
    member_2_: f32,
    member_3_: f32,
    member_4_: f32,
    member_5_: f32,
}

struct type_33_ {
    member: i32,
    member_1_: i32,
}

struct type_34_ {
    member: u32,
    member_1_: i32,
}

struct type_37_ {
    member: type_30_,
    member_1_: type_30_,
    member_2_: type_30_,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
var<private> num_wg_1_: vec3<u32>;
@group(0) @binding(0) 
var<uniform> uniforms: type_18_;
@group(0) @binding(1) 
var<storage> core_inputs: type_7_;
@group(0) @binding(2) 
var<storage> quad_params: type_7_;
var<workgroup> scan_scratch: array<f32, 1536>;
@group(0) @binding(3) 
var<storage, read_write> tile_checkpoints: type_7_;
@group(0) @binding(4) 
var<storage, read_write> feature_outputs: type_7_;
var<workgroup> aux_scratch: array<f32, 256>;
var<workgroup> lane_buf: array<f32, 2048>;

fn function_() {
    var local: array<f32, 4>;
    var local_1_: array<f32, 4>;
    var local_2_: array<f32, 4>;
    var local_3_: array<f32, 4>;
    var local_4_: array<f32, 4>;
    var local_5_: array<f32, 4>;
    var local_6_: array<f32, 4>;
    var local_7_: array<f32, 4>;
    var local_8_: array<array<u32, 4>, 4>;
    var local_9_: array<f32, 4>;
    var local_10_: array<f32, 4>;
    var local_11_: array<f32, 4>;
    var local_12_: array<f32, 4>;
    var local_13_: array<f32, 4>;
    var local_14_: array<f32, 4>;
    var phi_296_: u32;
    var phi_297_: u32;
    var phi_314_: bool;
    var phi_322_: bool;
    var phi_332_: u32;
    var phi_363_: u32;
    var phi_374_: u32;
    var phi_377_: u32;
    var phi_375_: u32;
    var phi_378_: u32;
    var phi_389_: u32;
    var phi_392_: u32;
    var phi_394_: u32;
    var phi_396_: u32;
    var phi_410_: u32;
    var phi_411_: u32;
    var phi_390_: u32;
    var phi_393_: u32;
    var phi_395_: u32;
    var phi_397_: u32;
    var local_15_: u32;
    var phi_415_: u32;
    var phi_416_: u32;
    var phi_430_: u32;
    var phi_431_: u32;
    var phi_445_: vec4<f32>;
    var phi_458_: vec4<f32>;
    var phi_471_: vec4<f32>;
    var phi_484_: vec4<f32>;
    var phi_497_: vec4<f32>;
    var phi_510_: vec4<f32>;
    var phi_523_: vec4<f32>;
    var phi_536_: vec4<f32>;
    var phi_549_: vec4<f32>;
    var phi_562_: vec4<f32>;
    var phi_575_: vec4<f32>;
    var phi_588_: vec4<f32>;
    var phi_601_: vec4<f32>;
    var phi_614_: vec4<f32>;
    var phi_627_: vec4<f32>;
    var phi_640_: vec4<f32>;
    var phi_653_: vec4<f32>;
    var phi_666_: vec4<f32>;
    var phi_679_: vec4<f32>;
    var phi_692_: vec4<f32>;
    var phi_705_: vec4<f32>;
    var phi_718_: vec4<f32>;
    var phi_731_: vec4<f32>;
    var phi_757_: vec4<f32>;
    var phi_784_: bool;
    var phi_790_: f32;
    var phi_791_: f32;
    var phi_798_: f32;
    var phi_824_: bool;
    var phi_830_: f32;
    var phi_831_: f32;
    var phi_838_: f32;
    var phi_864_: bool;
    var phi_870_: f32;
    var phi_871_: f32;
    var phi_878_: f32;
    var phi_904_: bool;
    var phi_910_: f32;
    var phi_911_: f32;
    var phi_918_: f32;
    var phi_954_: vec4<f32>;
    var phi_966_: bool;
    var phi_983_: bool;
    var phi_1000_: bool;
    var phi_1017_: bool;
    var phi_1041_: vec4<f32>;
    var phi_1053_: bool;
    var phi_1070_: bool;
    var phi_1087_: bool;
    var phi_1104_: bool;
    var phi_1128_: vec4<f32>;
    var phi_1140_: bool;
    var phi_1157_: bool;
    var phi_1174_: bool;
    var phi_1191_: bool;
    var phi_1215_: vec4<f32>;
    var phi_1227_: bool;
    var phi_1244_: bool;
    var phi_1261_: bool;
    var phi_1278_: bool;
    var phi_1302_: vec4<f32>;
    var phi_1314_: bool;
    var phi_1331_: bool;
    var phi_1348_: bool;
    var phi_1365_: bool;
    var phi_1389_: vec4<f32>;
    var phi_1401_: bool;
    var phi_1418_: bool;
    var phi_1435_: bool;
    var phi_1452_: bool;
    var phi_1476_: vec4<f32>;
    var phi_1488_: bool;
    var phi_1505_: f32;
    var phi_1517_: bool;
    var phi_1534_: f32;
    var phi_1546_: bool;
    var phi_1563_: f32;
    var phi_1575_: bool;
    var phi_1592_: f32;
    var phi_1611_: vec4<f32>;
    var phi_1623_: bool;
    var phi_1640_: f32;
    var phi_1652_: bool;
    var phi_1669_: f32;
    var phi_1681_: bool;
    var phi_1698_: f32;
    var phi_1710_: bool;
    var phi_1727_: f32;
    var phi_1736_: type_15_;
    var phi_1737_: type_15_;
    var phi_1760_: type_15_;
    var phi_34691_: bool;
    var phi_34863_: bool;
    var phi_34862_: bool;
    var phi_1880_: bool;
    var phi_34861_: bool;
    var phi_1882_: bool;
    var phi_1886_: bool;
    var phi_1887_: bool;
    var phi_34860_: bool;
    var phi_1888_: bool;
    var phi_1891_: bool;
    var phi_1892_: bool;
    var phi_1893_: bool;
    var phi_1903_: bool;
    var phi_1910_: type_15_;
    var phi_1911_: type_15_;
    var phi_1934_: type_15_;
    var phi_34855_: bool;
    var phi_34894_: bool;
    var phi_34891_: bool;
    var phi_1960_: type_15_;
    var phi_1961_: type_15_;
    var phi_1984_: type_15_;
    var phi_2005_: bool;
    var phi_2086_: bool;
    var phi_2102_: bool;
    var phi_2118_: bool;
    var phi_2134_: bool;
    var phi_2162_: bool;
    var phi_2178_: bool;
    var phi_2194_: bool;
    var phi_2210_: bool;
    var phi_2238_: bool;
    var phi_2254_: bool;
    var phi_2270_: bool;
    var phi_2286_: bool;
    var phi_2314_: bool;
    var phi_2330_: bool;
    var phi_2346_: bool;
    var phi_2362_: bool;
    var phi_2390_: bool;
    var phi_2406_: bool;
    var phi_2422_: bool;
    var phi_2438_: bool;
    var phi_2466_: bool;
    var phi_2482_: bool;
    var phi_2498_: bool;
    var phi_2514_: bool;
    var phi_2542_: bool;
    var phi_2558_: bool;
    var phi_2574_: bool;
    var phi_2590_: bool;
    var phi_2618_: bool;
    var phi_2634_: bool;
    var phi_2650_: bool;
    var phi_2666_: bool;
    var phi_34888_: bool;
    var phi_2685_: type_15_;
    var phi_2686_: type_15_;
    var phi_2709_: type_15_;
    var phi_2724_: u32;
    var phi_2753_: vec4<f32>;
    var phi_2766_: vec4<f32>;
    var phi_2767_: vec4<f32>;
    var phi_2768_: vec4<f32>;
    var phi_34883_: bool;
    var phi_6574_: type_15_;
    var phi_6575_: type_15_;
    var phi_6598_: type_15_;
    var phi_6693_: f32;
    var phi_6704_: f32;
    var phi_6715_: f32;
    var phi_6726_: f32;
    var phi_6737_: f32;
    var phi_6748_: f32;
    var phi_6759_: f32;
    var phi_6770_: f32;
    var phi_6781_: f32;
    var phi_6792_: f32;
    var phi_7100_: f32;
    var phi_7111_: bool;
    var phi_7117_: f32;
    var phi_7131_: type_31_;
    var phi_7136_: u32;
    var phi_7142_: u32;
    var phi_7220_: u32;
    var phi_7223_: type_31_;
    var phi_7325_: type_31_;
    var phi_7221_: u32;
    var phi_7224_: type_31_;
    var phi_34864_: bool;
    var phi_7449_: array<f32, 2>;
    var phi_7461_: u32;
    var phi_7525_: array<f32, 2>;
    var phi_7780_: bool;
    var phi_7803_: bool;
    var phi_7831_: bool;
    var phi_7855_: bool;
    var phi_34978_: bool;
    var phi_34977_: bool;
    var phi_5578_: type_15_;
    var phi_5579_: type_15_;
    var phi_5602_: type_15_;
    var phi_5620_: f32;
    var phi_5631_: bool;
    var phi_5656_: f32;
    var phi_5667_: bool;
    var phi_34994_: bool;
    var phi_35021_: bool;
    var phi_5795_: f32;
    var phi_5812_: f32;
    var phi_5823_: f32;
    var phi_5834_: f32;
    var phi_5845_: f32;
    var phi_5856_: f32;
    var phi_6071_: f32;
    var phi_6146_: f32;
    var phi_6151_: u32;
    var phi_6157_: u32;
    var phi_6182_: type_15_;
    var phi_6185_: f32;
    var phi_6187_: f32;
    var phi_6183_: type_15_;
    var phi_6210_: type_15_;
    var phi_23631_: bool;
    var phi_23691_: bool;
    var phi_23751_: bool;
    var phi_23811_: bool;
    var phi_6377_: u32;
    var phi_6378_: u32;
    var phi_6429_: u32;
    var phi_6430_: u32;
    var phi_6186_: f32;
    var phi_6188_: f32;
    var phi_35006_: bool;
    var local_16_: f32;
    var phi_6538_: bool;
    var local_17_: f32;
    var phi_6560_: bool;
    var phi_35313_: bool;
    var phi_35084_: bool;
    var phi_2777_: type_15_;
    var phi_2778_: type_15_;
    var phi_2801_: type_15_;
    var phi_2914_: f32;
    var phi_2931_: f32;
    var phi_2942_: f32;
    var phi_2953_: f32;
    var phi_2964_: f32;
    var phi_2975_: f32;
    var phi_3190_: f32;
    var phi_3265_: f32;
    var phi_3283_: f32;
    var phi_3294_: bool;
    var phi_3300_: f32;
    var phi_3309_: u32;
    var phi_35072_: bool;
    var phi_3328_: type_15_;
    var phi_3331_: array<f32, 2>;
    var phi_3333_: f32;
    var phi_3335_: f32;
    var phi_3337_: array<f32, 2>;
    var phi_3339_: bool;
    var phi_3329_: type_15_;
    var phi_3362_: type_15_;
    var phi_23871_: bool;
    var phi_23886_: bool;
    var phi_23901_: bool;
    var phi_23916_: bool;
    var phi_3410_: type_31_;
    var phi_3473_: u32;
    var phi_3476_: type_31_;
    var phi_3578_: type_31_;
    var phi_3474_: u32;
    var phi_3477_: type_31_;
    var phi_35058_: bool;
    var phi_3702_: array<f32, 2>;
    var phi_3714_: u32;
    var phi_3778_: array<f32, 2>;
    var phi_23931_: bool;
    var phi_3893_: bool;
    var phi_23991_: bool;
    var phi_3858_: bool;
    var phi_3899_: f32;
    var phi_3900_: f32;
    var phi_3901_: f32;
    var phi_3902_: bool;
    var phi_3908_: f32;
    var phi_3910_: f32;
    var phi_3911_: bool;
    var phi_24051_: bool;
    var phi_4027_: bool;
    var phi_24111_: bool;
    var phi_3992_: bool;
    var phi_4033_: f32;
    var phi_4034_: f32;
    var phi_4035_: f32;
    var phi_4036_: bool;
    var phi_4042_: f32;
    var phi_4044_: f32;
    var phi_4045_: bool;
    var phi_24171_: bool;
    var phi_4177_: bool;
    var phi_24231_: bool;
    var phi_4142_: bool;
    var phi_4183_: f32;
    var phi_4184_: f32;
    var phi_4185_: f32;
    var phi_4186_: bool;
    var phi_4192_: f32;
    var phi_4194_: f32;
    var phi_4195_: bool;
    var phi_24291_: bool;
    var phi_4313_: bool;
    var phi_24351_: bool;
    var phi_4278_: bool;
    var phi_4319_: f32;
    var phi_4320_: f32;
    var phi_4321_: f32;
    var phi_4322_: bool;
    var phi_4328_: f32;
    var phi_4330_: f32;
    var phi_4331_: bool;
    var phi_4346_: f32;
    var phi_4348_: f32;
    var phi_4349_: bool;
    var phi_4354_: f32;
    var phi_4392_: u32;
    var phi_4393_: u32;
    var phi_35090_: bool;
    var phi_4453_: bool;
    var phi_4458_: bool;
    var phi_35183_: bool;
    var phi_4459_: array<f32, 2>;
    var phi_4460_: f32;
    var phi_4461_: f32;
    var phi_4462_: array<f32, 2>;
    var phi_4463_: bool;
    var phi_35182_: bool;
    var phi_3332_: array<f32, 2>;
    var phi_3334_: f32;
    var phi_3336_: f32;
    var phi_3338_: array<f32, 2>;
    var phi_3340_: bool;
    var phi_35181_: bool;
    var local_18_: bool;
    var phi_4826_: type_15_;
    var phi_4829_: f32;
    var phi_4831_: f32;
    var phi_4827_: type_15_;
    var phi_4854_: type_15_;
    var phi_24411_: bool;
    var phi_24471_: bool;
    var phi_24531_: bool;
    var phi_24591_: bool;
    var phi_5020_: u32;
    var phi_5021_: u32;
    var phi_5072_: u32;
    var phi_5073_: u32;
    var phi_4830_: f32;
    var phi_4832_: f32;
    var phi_35187_: bool;
    var local_19_: f32;
    var local_20_: f32;
    var phi_35234_: bool;
    var phi_5205_: type_15_;
    var phi_5208_: f32;
    var phi_5210_: f32;
    var phi_5206_: type_15_;
    var phi_5233_: type_15_;
    var phi_24651_: bool;
    var phi_24711_: bool;
    var phi_24771_: bool;
    var phi_24831_: bool;
    var phi_5400_: u32;
    var phi_5401_: u32;
    var phi_5452_: u32;
    var phi_5453_: u32;
    var phi_5209_: f32;
    var phi_5211_: f32;
    var phi_35230_: bool;
    var local_21_: f32;
    var local_22_: f32;
    var phi_35274_: bool;
    var phi_24891_: bool;
    var phi_24906_: bool;
    var phi_24921_: bool;
    var local_23_: f32;
    var local_24_: f32;
    var phi_24936_: bool;
    var local_25_: f32;
    var local_26_: f32;
    var local_27_: f32;
    var local_28_: f32;
    var local_29_: array<f32, 2>;
    var local_30_: array<f32, 2>;
    var local_31_: array<f32, 2>;
    var local_32_: array<f32, 2>;
    var phi_35273_: bool;
    var phi_35272_: bool;
    var phi_35271_: bool;
    var phi_35312_: bool;
    var phi_35311_: bool;
    var phi_16108_: f32;
    var phi_16119_: f32;
    var phi_16130_: f32;
    var phi_16141_: f32;
    var phi_16152_: f32;
    var phi_16163_: f32;
    var phi_16259_: bool;
    var phi_16265_: f32;
    var phi_16266_: f32;
    var phi_16273_: f32;
    var phi_16298_: bool;
    var phi_16304_: f32;
    var phi_16305_: f32;
    var phi_16312_: f32;
    var phi_16337_: bool;
    var phi_16343_: f32;
    var phi_16344_: f32;
    var phi_16351_: f32;
    var phi_16376_: bool;
    var phi_16382_: f32;
    var phi_16383_: f32;
    var phi_16390_: f32;
    var phi_16415_: bool;
    var phi_16421_: f32;
    var phi_16422_: f32;
    var phi_16429_: f32;
    var phi_16454_: bool;
    var phi_16460_: f32;
    var phi_16461_: f32;
    var phi_16468_: f32;
    var phi_16539_: f32;
    var phi_16550_: f32;
    var phi_16645_: type_15_;
    var phi_16648_: f32;
    var phi_16650_: f32;
    var phi_16646_: type_15_;
    var phi_16673_: type_15_;
    var phi_16743_: u32;
    var phi_16744_: u32;
    var phi_24951_: bool;
    var phi_25011_: bool;
    var phi_25071_: bool;
    var phi_25131_: bool;
    var phi_25191_: bool;
    var phi_25251_: bool;
    var phi_25311_: bool;
    var phi_25371_: bool;
    var phi_17485_: u32;
    var phi_17486_: u32;
    var phi_17547_: u32;
    var phi_17548_: u32;
    var phi_25551_: bool;
    var phi_25611_: bool;
    var phi_25671_: bool;
    var phi_25731_: bool;
    var phi_25791_: bool;
    var phi_25851_: bool;
    var phi_25911_: bool;
    var phi_25971_: bool;
    var phi_18289_: u32;
    var phi_18290_: u32;
    var phi_16649_: f32;
    var phi_16651_: f32;
    var phi_35287_: bool;
    var local_33_: f32;
    var phi_18421_: bool;
    var local_34_: f32;
    var phi_18444_: bool;
    var phi_36592_: bool;
    var phi_35650_: bool;
    var phi_7873_: type_15_;
    var phi_7874_: type_15_;
    var phi_7897_: type_15_;
    var phi_7915_: f32;
    var phi_7926_: f32;
    var phi_7937_: f32;
    var phi_7948_: f32;
    var phi_7959_: f32;
    var phi_7970_: f32;
    var phi_8066_: bool;
    var phi_8072_: f32;
    var phi_8073_: f32;
    var phi_8080_: f32;
    var phi_8105_: bool;
    var phi_8111_: f32;
    var phi_8112_: f32;
    var phi_8119_: f32;
    var phi_8144_: bool;
    var phi_8150_: f32;
    var phi_8151_: f32;
    var phi_8158_: f32;
    var phi_8183_: bool;
    var phi_8189_: f32;
    var phi_8190_: f32;
    var phi_8197_: f32;
    var phi_8222_: bool;
    var phi_8228_: f32;
    var phi_8229_: f32;
    var phi_8236_: f32;
    var phi_8261_: bool;
    var phi_8267_: f32;
    var phi_8268_: f32;
    var phi_8275_: f32;
    var phi_8346_: f32;
    var phi_8357_: f32;
    var phi_8445_: f32;
    var phi_8465_: f32;
    var phi_35623_: bool;
    var phi_8496_: type_15_;
    var phi_8499_: f32;
    var phi_8501_: bool;
    var phi_8497_: type_15_;
    var phi_8524_: type_15_;
    var phi_8547_: f32;
    var phi_8576_: u32;
    var phi_8577_: u32;
    var phi_26151_: bool;
    var phi_26211_: bool;
    var phi_26271_: bool;
    var phi_26331_: bool;
    var phi_26391_: bool;
    var phi_26451_: bool;
    var phi_26511_: bool;
    var phi_26571_: bool;
    var phi_9318_: u32;
    var phi_9319_: u32;
    var phi_9363_: bool;
    var phi_9379_: bool;
    var phi_9395_: bool;
    var phi_9412_: bool;
    var phi_9424_: f32;
    var phi_9436_: f32;
    var phi_26751_: bool;
    var phi_26811_: bool;
    var phi_26931_: bool;
    var phi_26991_: bool;
    var phi_27051_: bool;
    var phi_9932_: f32;
    var phi_9940_: f32;
    var phi_10069_: bool;
    var phi_10075_: f32;
    var phi_10076_: f32;
    var phi_9995_: f32;
    var phi_9997_: f32;
    var phi_9998_: bool;
    var phi_10018_: f32;
    var phi_10077_: f32;
    var phi_10080_: type_28_;
    var phi_27111_: bool;
    var phi_10197_: bool;
    var phi_27171_: bool;
    var phi_10162_: bool;
    var phi_10203_: f32;
    var phi_10204_: f32;
    var phi_10205_: f32;
    var phi_10206_: bool;
    var phi_10212_: f32;
    var phi_10214_: f32;
    var phi_10215_: bool;
    var phi_10220_: f32;
    var phi_10258_: u32;
    var phi_10259_: u32;
    var phi_35614_: bool;
    var phi_10318_: type_28_;
    var phi_10346_: u32;
    var phi_10349_: type_28_;
    var phi_10389_: type_28_;
    var phi_10347_: u32;
    var phi_10350_: type_28_;
    var phi_35904_: bool;
    var local_35_: type_28_;
    var local_36_: type_28_;
    var phi_10425_: u32;
    var phi_35919_: bool;
    var phi_10447_: type_33_;
    var phi_10450_: f32;
    var phi_10452_: f32;
    var phi_10454_: bool;
    var phi_10456_: bool;
    var phi_10483_: type_34_;
    var phi_10448_: type_33_;
    var phi_10497_: type_34_;
    var phi_10629_: bool;
    var phi_10635_: f32;
    var phi_10636_: f32;
    var phi_10555_: f32;
    var phi_10557_: f32;
    var phi_10558_: bool;
    var phi_10578_: f32;
    var phi_10637_: f32;
    var phi_10638_: f32;
    var phi_10659_: f32;
    var phi_10688_: u32;
    var phi_10689_: u32;
    var phi_27231_: bool;
    var phi_27291_: bool;
    var phi_27351_: bool;
    var phi_27411_: bool;
    var phi_27471_: bool;
    var phi_27531_: bool;
    var phi_27591_: bool;
    var phi_27651_: bool;
    var phi_11430_: u32;
    var phi_11431_: u32;
    var phi_28191_: bool;
    var phi_12265_: bool;
    var phi_28251_: bool;
    var phi_12230_: bool;
    var phi_12271_: f32;
    var phi_12272_: f32;
    var phi_12273_: f32;
    var phi_12274_: bool;
    var phi_12280_: f32;
    var phi_12282_: f32;
    var phi_12283_: bool;
    var phi_12288_: f32;
    var phi_12326_: u32;
    var phi_12327_: u32;
    var phi_35910_: bool;
    var phi_12383_: bool;
    var phi_12401_: bool;
    var phi_36060_: bool;
    var phi_12403_: f32;
    var phi_12404_: f32;
    var phi_12405_: bool;
    var phi_12406_: bool;
    var phi_36059_: bool;
    var phi_10451_: f32;
    var phi_10453_: f32;
    var phi_10455_: bool;
    var phi_10457_: bool;
    var phi_36058_: bool;
    var phi_36067_: bool;
    var phi_12407_: f32;
    var phi_12408_: bool;
    var phi_36066_: bool;
    var phi_12409_: f32;
    var phi_12410_: bool;
    var phi_36065_: bool;
    var phi_8500_: f32;
    var phi_8502_: bool;
    var phi_36064_: bool;
    var phi_12424_: f32;
    var local_37_: f32;
    var phi_12453_: u32;
    var phi_12454_: u32;
    var phi_28311_: bool;
    var phi_28371_: bool;
    var phi_28431_: bool;
    var phi_28491_: bool;
    var phi_28551_: bool;
    var phi_28611_: bool;
    var phi_28671_: bool;
    var phi_28731_: bool;
    var phi_13195_: u32;
    var phi_13196_: u32;
    var phi_13240_: bool;
    var phi_13256_: bool;
    var phi_13272_: bool;
    var phi_13289_: bool;
    var phi_13301_: f32;
    var phi_13313_: f32;
    var phi_28911_: bool;
    var phi_28971_: bool;
    var phi_29091_: bool;
    var phi_29151_: bool;
    var phi_29211_: bool;
    var phi_13809_: f32;
    var phi_13817_: f32;
    var phi_13946_: bool;
    var phi_13952_: f32;
    var phi_13953_: f32;
    var phi_13872_: f32;
    var phi_13874_: f32;
    var phi_13875_: bool;
    var phi_13895_: f32;
    var phi_13954_: f32;
    var phi_13957_: type_28_;
    var local_38_: f32;
    var local_39_: f32;
    var local_40_: f32;
    var phi_29271_: bool;
    var phi_14073_: bool;
    var phi_29331_: bool;
    var phi_14038_: bool;
    var phi_14079_: f32;
    var phi_14080_: f32;
    var phi_14081_: f32;
    var phi_14082_: bool;
    var phi_14088_: f32;
    var phi_14090_: f32;
    var phi_14091_: bool;
    var phi_14096_: f32;
    var phi_14134_: u32;
    var phi_14135_: u32;
    var phi_36068_: bool;
    var phi_14221_: type_15_;
    var phi_14224_: f32;
    var phi_14226_: f32;
    var phi_14222_: type_15_;
    var phi_14249_: type_15_;
    var phi_14319_: u32;
    var phi_14320_: u32;
    var phi_29391_: bool;
    var phi_29451_: bool;
    var phi_29511_: bool;
    var phi_29571_: bool;
    var phi_29631_: bool;
    var phi_29691_: bool;
    var phi_29751_: bool;
    var phi_29811_: bool;
    var phi_15061_: u32;
    var phi_15062_: u32;
    var phi_15123_: u32;
    var phi_15124_: u32;
    var phi_29991_: bool;
    var phi_30051_: bool;
    var phi_30111_: bool;
    var phi_30171_: bool;
    var phi_30231_: bool;
    var phi_30291_: bool;
    var phi_30351_: bool;
    var phi_30411_: bool;
    var phi_15865_: u32;
    var phi_15866_: u32;
    var phi_14225_: f32;
    var phi_14227_: f32;
    var phi_36303_: bool;
    var local_41_: f32;
    var local_42_: f32;
    var phi_36586_: bool;
    var phi_36581_: bool;
    var phi_36580_: bool;
    var phi_36579_: bool;
    var phi_36591_: bool;
    var phi_18466_: type_15_;
    var phi_18467_: type_15_;
    var phi_18490_: type_15_;
    var phi_18861_: f32;
    var phi_18872_: f32;
    var phi_18883_: f32;
    var phi_30591_: bool;
    var phi_30606_: bool;
    var phi_19010_: bool;
    var phi_19030_: bool;
    var phi_36587_: bool;
    var phi_36613_: bool;
    var phi_36612_: bool;
    var phi_19141_: type_15_;
    var phi_19142_: type_15_;
    var phi_19165_: type_15_;
    var phi_36608_: bool;
    var phi_19078_: type_15_;
    var phi_19079_: type_15_;
    var phi_19102_: type_15_;
    var phi_36614_: bool;
    var phi_37437_: bool;
    var phi_30621_: bool;
    var phi_30636_: bool;
    var phi_30651_: bool;
    var phi_30666_: bool;
    var phi_30681_: bool;
    var phi_30696_: bool;
    var phi_30711_: bool;
    var phi_30726_: bool;
    var phi_30741_: bool;
    var phi_30756_: bool;
    var phi_30771_: bool;
    var phi_30786_: bool;
    var phi_30801_: bool;
    var phi_30816_: bool;
    var phi_30831_: bool;
    var phi_30846_: bool;
    var phi_30861_: bool;
    var phi_30876_: bool;
    var phi_30891_: bool;
    var phi_30906_: bool;
    var phi_30921_: bool;
    var phi_30936_: bool;
    var phi_30951_: bool;
    var phi_30966_: bool;
    var phi_30981_: bool;
    var phi_30996_: bool;
    var phi_31011_: bool;
    var phi_31026_: bool;
    var phi_31041_: bool;
    var phi_31056_: bool;
    var phi_31071_: bool;
    var phi_31086_: bool;
    var phi_31101_: bool;
    var phi_31116_: bool;
    var phi_31131_: bool;
    var phi_31146_: bool;
    var phi_31161_: bool;
    var phi_31176_: bool;
    var phi_31191_: bool;
    var phi_31206_: bool;
    var phi_31221_: bool;
    var phi_31236_: bool;
    var phi_31251_: bool;
    var phi_31266_: bool;
    var phi_31281_: bool;
    var phi_31296_: bool;
    var phi_31311_: bool;
    var phi_31326_: bool;
    var phi_31341_: bool;
    var phi_31356_: bool;
    var phi_31371_: bool;
    var phi_31386_: bool;
    var phi_31401_: bool;
    var phi_31416_: bool;
    var phi_31431_: bool;
    var phi_31446_: bool;
    var phi_31461_: bool;
    var phi_31476_: bool;
    var phi_31491_: bool;
    var phi_31506_: bool;
    var phi_31521_: bool;
    var phi_31536_: bool;
    var phi_31551_: bool;
    var phi_31566_: bool;
    var phi_19670_: bool;
    var phi_19672_: bool;
    var phi_19673_: bool;
    var phi_19674_: bool;
    var phi_19675_: bool;
    var phi_31581_: bool;
    var phi_31596_: bool;
    var phi_31611_: bool;
    var phi_31626_: bool;
    var phi_20053_: type_37_;
    var phi_31641_: bool;
    var phi_31656_: bool;
    var phi_31671_: bool;
    var phi_31686_: bool;
    var phi_31701_: bool;
    var phi_31716_: bool;
    var phi_31731_: bool;
    var phi_31746_: bool;
    var phi_31761_: bool;
    var phi_31776_: bool;
    var phi_31791_: bool;
    var phi_31806_: bool;
    var phi_31821_: bool;
    var phi_31836_: bool;
    var phi_31851_: bool;
    var phi_31866_: bool;
    var phi_31881_: bool;
    var phi_31896_: bool;
    var phi_31911_: bool;
    var phi_31926_: bool;
    var phi_31941_: bool;
    var phi_31956_: bool;
    var phi_31971_: bool;
    var phi_31986_: bool;
    var phi_32001_: bool;
    var phi_32016_: bool;
    var phi_32031_: bool;
    var phi_32046_: bool;
    var phi_32061_: bool;
    var phi_32076_: bool;
    var phi_32091_: bool;
    var phi_32106_: bool;
    var phi_32121_: bool;
    var phi_32136_: bool;
    var phi_32151_: bool;
    var phi_32166_: bool;
    var phi_32181_: bool;
    var phi_32196_: bool;
    var phi_32211_: bool;
    var phi_32226_: bool;
    var phi_32241_: bool;
    var phi_32256_: bool;
    var phi_32271_: bool;
    var phi_32286_: bool;
    var phi_32301_: bool;
    var phi_32316_: bool;
    var phi_32331_: bool;
    var phi_32346_: bool;
    var phi_32361_: bool;
    var phi_32376_: bool;
    var phi_32391_: bool;
    var phi_32406_: bool;
    var phi_32421_: bool;
    var phi_32436_: bool;
    var phi_32451_: bool;
    var phi_32466_: bool;
    var phi_32481_: bool;
    var phi_32496_: bool;
    var phi_32511_: bool;
    var phi_32526_: bool;
    var phi_32541_: bool;
    var phi_32556_: bool;
    var phi_32571_: bool;
    var phi_32586_: bool;
    var phi_20499_: bool;
    var phi_20501_: bool;
    var phi_20502_: bool;
    var phi_20503_: bool;
    var phi_20504_: bool;
    var phi_32601_: bool;
    var phi_32616_: bool;
    var phi_32631_: bool;
    var phi_32646_: bool;
    var phi_20882_: type_37_;
    var phi_20899_: bool;
    var phi_20901_: bool;
    var phi_20902_: bool;
    var phi_20903_: bool;
    var phi_20904_: bool;
    var phi_32661_: bool;
    var phi_32676_: bool;
    var phi_32691_: bool;
    var phi_32706_: bool;
    var phi_32721_: bool;
    var phi_32736_: bool;
    var phi_32751_: bool;
    var phi_32766_: bool;
    var phi_32781_: bool;
    var phi_32796_: bool;
    var phi_32811_: bool;
    var phi_32826_: bool;
    var phi_32841_: bool;
    var phi_32856_: bool;
    var phi_32871_: bool;
    var phi_32886_: bool;
    var phi_32901_: bool;
    var phi_32916_: bool;
    var phi_32931_: bool;
    var phi_32946_: bool;
    var phi_32961_: bool;
    var phi_32976_: bool;
    var phi_32991_: bool;
    var phi_33006_: bool;
    var phi_33021_: bool;
    var phi_33036_: bool;
    var phi_33051_: bool;
    var phi_33066_: bool;
    var phi_33081_: bool;
    var phi_33096_: bool;
    var phi_33111_: bool;
    var phi_33126_: bool;
    var phi_33141_: bool;
    var phi_33156_: bool;
    var phi_33171_: bool;
    var phi_33186_: bool;
    var phi_33201_: bool;
    var phi_33216_: bool;
    var phi_33231_: bool;
    var phi_33246_: bool;
    var phi_33261_: bool;
    var phi_33276_: bool;
    var phi_33291_: bool;
    var phi_33306_: bool;
    var phi_33561_: bool;
    var phi_33576_: bool;
    var phi_33591_: bool;
    var phi_33606_: bool;
    var phi_33621_: bool;
    var phi_33636_: bool;
    var phi_33651_: bool;
    var phi_33666_: bool;
    var phi_33681_: bool;
    var phi_33696_: bool;
    var phi_33711_: bool;
    var phi_33726_: bool;
    var phi_33741_: bool;
    var phi_33756_: bool;
    var phi_33771_: bool;
    var phi_33786_: bool;
    var phi_33801_: bool;
    var phi_33816_: bool;
    var phi_33831_: bool;
    var phi_33846_: bool;
    var phi_33861_: bool;
    var phi_33876_: bool;
    var phi_33891_: bool;
    var phi_33906_: bool;
    var phi_33921_: bool;
    var phi_33936_: bool;
    var phi_33951_: bool;
    var phi_33966_: bool;
    var phi_33981_: bool;
    var phi_33996_: bool;
    var phi_34011_: bool;
    var phi_34026_: bool;
    var phi_34041_: bool;
    var phi_34056_: bool;
    var phi_34071_: bool;
    var phi_34086_: bool;
    var phi_34101_: bool;
    var phi_34116_: bool;
    var phi_34131_: bool;
    var phi_34146_: bool;
    var phi_34161_: bool;
    var phi_34176_: bool;
    var phi_34191_: bool;
    var phi_34206_: bool;
    var phi_34221_: bool;
    var phi_34236_: bool;
    var phi_34251_: bool;
    var phi_34266_: bool;
    var phi_34281_: bool;
    var phi_34296_: bool;
    var phi_34311_: bool;
    var phi_34326_: bool;
    var phi_34581_: bool;
    var phi_34596_: bool;
    var phi_34611_: bool;
    var phi_34626_: bool;
    var phi_34641_: bool;
    var phi_34656_: bool;
    var phi_34671_: bool;
    var phi_34686_: bool;
    var phi_22812_: type_30_;
    var phi_22813_: type_30_;
    var phi_22969_: bool;
    var phi_22985_: bool;
    var phi_23001_: bool;
    var phi_23017_: bool;
    var phi_23055_: bool;
    var phi_23071_: bool;
    var phi_23087_: bool;
    var phi_23103_: bool;
    var phi_23135_: bool;
    var phi_23151_: bool;
    var phi_23167_: bool;
    var phi_23183_: bool;
    var phi_23215_: bool;
    var phi_23231_: bool;
    var phi_23247_: bool;
    var phi_23263_: bool;
    var phi_23295_: bool;
    var phi_23311_: bool;
    var phi_23327_: bool;
    var phi_23343_: bool;
    var phi_23375_: bool;
    var phi_23391_: bool;
    var phi_23407_: bool;
    var phi_23423_: bool;
    var phi_23447_: type_15_;
    var phi_23448_: type_15_;
    var phi_23471_: type_15_;
    var phi_23496_: bool;
    var phi_23513_: f32;
    var phi_23538_: bool;
    var phi_23555_: f32;
    var phi_36618_: bool;
    var phi_37457_: bool;
    var phi_37456_: bool;
    var phi_37455_: bool;
    var phi_37454_: bool;
    var phi_37453_: bool;
    var phi_37452_: bool;
    var phi_37459_: bool;
    var phi_37458_: bool;
    var local_43_: u32;
    var local_44_: u32;
    var local_45_: f32;
    var local_46_: bool;

    switch bitcast<i32>(0u) {
        default: {
            let _e146_ = wg_id_1_;
            let _e147_ = num_wg_1_;
            let _e148_ = lid_1_;
            let _e150_ = arrayLength((&core_inputs.member));
            let _e152_ = arrayLength((&quad_params.member));
            let _e154_ = arrayLength((&tile_checkpoints.member));
            let _e156_ = arrayLength((&feature_outputs.member));
            let _e157_ = (_e148_ & 255u);
            let _e158_ = (_e157_ & 127u);
            let _e161_ = ((_e157_ >> bitcast<u32>(7i)) != 0u);
            let _e166_ = select(1u, _e147_.y, (1u < _e147_.y));
            let _e170_ = uniforms.member.member_2_;
            let _e172_ = select(1u, _e170_, (1u < _e170_));
            let _e175_ = uniforms.member.member_7_;
            let _e176_ = (_e175_ == 4294967295u);
            if _e176_ {
                phi_297_ = select(16u, _e146_.x, (_e146_.x < _e172_));
            } else {
                let _e177_ = (_e146_.x < 4u);
                if _e177_ {
                    local_8_ = array<array<u32, 4>, 4>(array<u32, 4>(0u, 4u, 6u, 8u), array<u32, 4>(1u, 5u, 9u, 16u), array<u32, 4>(2u, 7u, 16u, 16u), array<u32, 4>(3u, 16u, 16u, 16u));
                    let _e178_ = (_e175_ & 3u);
                    if (_e178_ < 4u) {
                    } else {
                        break;
                    }
                    if _e177_ {
                    } else {
                        break;
                    }
                    let _e182_ = local_8_[_e178_][_e146_.x];
                    phi_296_ = select(16u, _e182_, (_e182_ < _e172_));
                } else {
                    phi_296_ = 16u;
                }
                let _e186_ = phi_296_;
                phi_297_ = _e186_;
            }
            let _e190_ = phi_297_;
            if (_e190_ >= _e172_) {
            } else {
                let _e193_ = ((_e146_.z * _e172_) + _e190_);
                if _e176_ {
                    phi_314_ = false;
                } else {
                    phi_314_ = ((_e175_ & 1073741824u) != 0u);
                }
                let _e197_ = phi_314_;
                if _e176_ {
                    phi_322_ = false;
                } else {
                    phi_322_ = ((_e175_ & 3u) == 0u);
                }
                let _e201_ = phi_322_;
                let _e204_ = uniforms.member.member;
                if (_e204_ > 0u) {
                    phi_332_ = (((_e204_ - 1u) >> bitcast<u32>(8i)) + 1u);
                } else {
                    phi_332_ = 0u;
                }
                let _e211_ = phi_332_;
                let _e214_ = uniforms.member.member_1_;
                let _e216_ = select(1u, _e214_, (1u < _e214_));
                let _e218_ = select(_e214_, _e211_, (_e211_ < _e214_));
                let _e221_ = uniforms.member.member_4_;
                let _e223_ = select(_e221_, _e218_, (_e218_ < _e221_));
                let _e224_ = (_e166_ <= 1u);
                if _e224_ {
                    phi_430_ = _e223_;
                    phi_431_ = _e218_;
                } else {
                    if (_e218_ > _e223_) {
                        phi_363_ = (_e218_ - _e223_);
                    } else {
                        phi_363_ = 0u;
                    }
                    let _e228_ = phi_363_;
                    if (_e228_ == 0u) {
                        phi_416_ = 0u;
                    } else {
                        if _e224_ {
                            phi_415_ = _e228_;
                        } else {
                            let _e230_ = (_e228_ - 1u);
                            phi_374_ = 1u;
                            phi_377_ = _e166_;
                            loop {
                                let _e232_ = phi_374_;
                                let _e234_ = phi_377_;
                                local_43_ = _e232_;
                                local_44_ = _e234_;
                                let _e237_ = (_e234_ <= (_e230_ >> bitcast<u32>(1i)));
                                if _e237_ {
                                    phi_375_ = (_e232_ << bitcast<u32>(1i));
                                    phi_378_ = (_e234_ << bitcast<u32>(1i));
                                } else {
                                    phi_375_ = u32();
                                    phi_378_ = u32();
                                }
                                let _e243_ = phi_375_;
                                let _e245_ = phi_378_;
                                phi_374_ = _e243_;
                                phi_377_ = _e245_;
                                if !(_e237_) {
                                    break;
                                }
                            }
                            phi_389_ = 0u;
                            let _e9186_ = local_43_;
                            phi_392_ = _e9186_;
                            phi_394_ = _e230_;
                            let _e9190_ = local_44_;
                            phi_396_ = _e9190_;
                            loop {
                                let _e248_ = phi_389_;
                                let _e250_ = phi_392_;
                                let _e252_ = phi_394_;
                                let _e254_ = phi_396_;
                                local_15_ = _e248_;
                                let _e255_ = (_e250_ > 0u);
                                if _e255_ {
                                    if (_e252_ >= _e254_) {
                                        phi_410_ = (_e248_ | _e250_);
                                        phi_411_ = (_e252_ - _e254_);
                                    } else {
                                        phi_410_ = _e248_;
                                        phi_411_ = _e252_;
                                    }
                                    let _e260_ = phi_410_;
                                    let _e262_ = phi_411_;
                                    phi_390_ = _e260_;
                                    phi_393_ = (_e250_ >> bitcast<u32>(1i));
                                    phi_395_ = _e262_;
                                    phi_397_ = (_e254_ >> bitcast<u32>(1i));
                                } else {
                                    phi_390_ = u32();
                                    phi_393_ = u32();
                                    phi_395_ = u32();
                                    phi_397_ = u32();
                                }
                                let _e268_ = phi_390_;
                                let _e270_ = phi_393_;
                                let _e272_ = phi_395_;
                                let _e274_ = phi_397_;
                                phi_389_ = _e268_;
                                phi_392_ = _e270_;
                                phi_394_ = _e272_;
                                phi_396_ = _e274_;
                                if !(_e255_) {
                                    break;
                                }
                            }
                            let _e277_ = local_15_;
                            phi_415_ = (_e277_ + 1u);
                        }
                        let _e280_ = phi_415_;
                        phi_416_ = _e280_;
                    }
                    let _e282_ = phi_416_;
                    let _e284_ = (_e223_ + (_e146_.y * _e282_));
                    let _e286_ = select(_e284_, _e218_, (_e218_ < _e284_));
                    let _e287_ = (_e286_ + _e282_);
                    phi_430_ = _e286_;
                    phi_431_ = select(_e287_, _e218_, (_e218_ < _e287_));
                }
                let _e291_ = phi_430_;
                let _e293_ = phi_431_;
                let _e294_ = (_e193_ * 25u);
                let _e295_ = (_e294_ < _e152_);
                if _e295_ {
                    if _e295_ {
                    } else {
                        break;
                    }
                    let _e298_ = quad_params.member[_e294_];
                    phi_445_ = _e298_;
                } else {
                    phi_445_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e300_ = phi_445_;
                let _e301_ = (_e294_ + 1u);
                let _e302_ = (_e301_ < _e152_);
                if _e302_ {
                    if _e302_ {
                    } else {
                        break;
                    }
                    let _e305_ = quad_params.member[_e301_];
                    phi_458_ = _e305_;
                } else {
                    phi_458_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e307_ = phi_458_;
                let _e308_ = (_e294_ + 2u);
                let _e309_ = (_e308_ < _e152_);
                if _e309_ {
                    if _e309_ {
                    } else {
                        break;
                    }
                    let _e312_ = quad_params.member[_e308_];
                    phi_471_ = _e312_;
                } else {
                    phi_471_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e314_ = phi_471_;
                let _e315_ = (_e294_ + 3u);
                let _e316_ = (_e315_ < _e152_);
                if _e316_ {
                    if _e316_ {
                    } else {
                        break;
                    }
                    let _e319_ = quad_params.member[_e315_];
                    phi_484_ = _e319_;
                } else {
                    phi_484_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e321_ = phi_484_;
                let _e322_ = (_e294_ + 4u);
                let _e323_ = (_e322_ < _e152_);
                if _e323_ {
                    if _e323_ {
                    } else {
                        break;
                    }
                    let _e326_ = quad_params.member[_e322_];
                    phi_497_ = _e326_;
                } else {
                    phi_497_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e328_ = phi_497_;
                let _e329_ = (_e294_ + 5u);
                let _e330_ = (_e329_ < _e152_);
                if _e330_ {
                    if _e330_ {
                    } else {
                        break;
                    }
                    let _e333_ = quad_params.member[_e329_];
                    phi_510_ = _e333_;
                } else {
                    phi_510_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e335_ = phi_510_;
                let _e336_ = (_e294_ + 6u);
                let _e337_ = (_e336_ < _e152_);
                if _e337_ {
                    if _e337_ {
                    } else {
                        break;
                    }
                    let _e340_ = quad_params.member[_e336_];
                    phi_523_ = _e340_;
                } else {
                    phi_523_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e342_ = phi_523_;
                let _e343_ = (_e294_ + 7u);
                let _e344_ = (_e343_ < _e152_);
                if _e344_ {
                    if _e344_ {
                    } else {
                        break;
                    }
                    let _e347_ = quad_params.member[_e343_];
                    phi_536_ = _e347_;
                } else {
                    phi_536_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e349_ = phi_536_;
                let _e350_ = (_e294_ + 8u);
                let _e351_ = (_e350_ < _e152_);
                if _e351_ {
                    if _e351_ {
                    } else {
                        break;
                    }
                    let _e354_ = quad_params.member[_e350_];
                    phi_549_ = _e354_;
                } else {
                    phi_549_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e356_ = phi_549_;
                let _e357_ = (_e294_ + 9u);
                let _e358_ = (_e357_ < _e152_);
                if _e358_ {
                    if _e358_ {
                    } else {
                        break;
                    }
                    let _e361_ = quad_params.member[_e357_];
                    phi_562_ = _e361_;
                } else {
                    phi_562_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e363_ = phi_562_;
                let _e364_ = (_e294_ + 10u);
                let _e365_ = (_e364_ < _e152_);
                if _e365_ {
                    if _e365_ {
                    } else {
                        break;
                    }
                    let _e368_ = quad_params.member[_e364_];
                    phi_575_ = _e368_;
                } else {
                    phi_575_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e370_ = phi_575_;
                let _e371_ = (_e294_ + 11u);
                let _e372_ = (_e371_ < _e152_);
                if _e372_ {
                    if _e372_ {
                    } else {
                        break;
                    }
                    let _e375_ = quad_params.member[_e371_];
                    phi_588_ = _e375_;
                } else {
                    phi_588_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e377_ = phi_588_;
                let _e378_ = (_e294_ + 12u);
                let _e379_ = (_e378_ < _e152_);
                if _e379_ {
                    if _e379_ {
                    } else {
                        break;
                    }
                    let _e382_ = quad_params.member[_e378_];
                    phi_601_ = _e382_;
                } else {
                    phi_601_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e384_ = phi_601_;
                let _e385_ = (_e294_ + 13u);
                let _e386_ = (_e385_ < _e152_);
                if _e386_ {
                    if _e386_ {
                    } else {
                        break;
                    }
                    let _e389_ = quad_params.member[_e385_];
                    phi_614_ = _e389_;
                } else {
                    phi_614_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e391_ = phi_614_;
                let _e392_ = (_e294_ + 14u);
                let _e393_ = (_e392_ < _e152_);
                if _e393_ {
                    if _e393_ {
                    } else {
                        break;
                    }
                    let _e396_ = quad_params.member[_e392_];
                    phi_627_ = _e396_;
                } else {
                    phi_627_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e398_ = phi_627_;
                let _e399_ = (_e294_ + 15u);
                let _e400_ = (_e399_ < _e152_);
                if _e400_ {
                    if _e400_ {
                    } else {
                        break;
                    }
                    let _e403_ = quad_params.member[_e399_];
                    phi_640_ = _e403_;
                } else {
                    phi_640_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e405_ = phi_640_;
                let _e406_ = (_e294_ + 16u);
                let _e407_ = (_e406_ < _e152_);
                if _e407_ {
                    if _e407_ {
                    } else {
                        break;
                    }
                    let _e410_ = quad_params.member[_e406_];
                    phi_653_ = _e410_;
                } else {
                    phi_653_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e412_ = phi_653_;
                let _e413_ = (_e294_ + 17u);
                let _e414_ = (_e413_ < _e152_);
                if _e414_ {
                    if _e414_ {
                    } else {
                        break;
                    }
                    let _e417_ = quad_params.member[_e413_];
                    phi_666_ = _e417_;
                } else {
                    phi_666_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e419_ = phi_666_;
                let _e420_ = (_e294_ + 18u);
                let _e421_ = (_e420_ < _e152_);
                if _e421_ {
                    if _e421_ {
                    } else {
                        break;
                    }
                    let _e424_ = quad_params.member[_e420_];
                    phi_679_ = _e424_;
                } else {
                    phi_679_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e426_ = phi_679_;
                let _e427_ = (_e294_ + 19u);
                let _e428_ = (_e427_ < _e152_);
                if _e428_ {
                    if _e428_ {
                    } else {
                        break;
                    }
                    let _e431_ = quad_params.member[_e427_];
                    phi_692_ = _e431_;
                } else {
                    phi_692_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e433_ = phi_692_;
                let _e434_ = (_e294_ + 20u);
                let _e435_ = (_e434_ < _e152_);
                if _e435_ {
                    if _e435_ {
                    } else {
                        break;
                    }
                    let _e438_ = quad_params.member[_e434_];
                    phi_705_ = _e438_;
                } else {
                    phi_705_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e440_ = phi_705_;
                let _e441_ = (_e294_ + 21u);
                let _e442_ = (_e441_ < _e152_);
                if _e442_ {
                    if _e442_ {
                    } else {
                        break;
                    }
                    let _e445_ = quad_params.member[_e441_];
                    phi_718_ = _e445_;
                } else {
                    phi_718_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e447_ = phi_718_;
                let _e448_ = (_e294_ + 22u);
                let _e449_ = (_e448_ < _e152_);
                if _e449_ {
                    if _e449_ {
                    } else {
                        break;
                    }
                    let _e452_ = quad_params.member[_e448_];
                    phi_731_ = _e452_;
                } else {
                    phi_731_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e454_ = phi_731_;
                let _e456_ = ((_e294_ + 23u) < _e152_);
                if _e456_ {
                    if _e456_ {
                    } else {
                        break;
                    }
                }
                let _e457_ = (_e294_ + 24u);
                let _e458_ = (_e457_ < _e152_);
                if _e458_ {
                    if _e458_ {
                    } else {
                        break;
                    }
                    let _e461_ = quad_params.member[_e457_];
                    phi_757_ = _e461_;
                } else {
                    phi_757_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e463_ = phi_757_;
                let _e465_ = bitcast<u32>(_e447_.x);
                let _e466_ = (_e465_ & 2139095040u);
                if (_e466_ != 2139095040u) {
                    if (_e447_.x <= 0f) {
                        phi_791_ = 0f;
                    } else {
                        if (_e447_.x >= 1f) {
                            phi_790_ = 1f;
                        } else {
                            if (_e466_ == 2139095040u) {
                                phi_784_ = true;
                            } else {
                                phi_784_ = (_e466_ == 0u);
                            }
                            let _e475_ = phi_784_;
                            phi_790_ = bitcast<f32>(select(_e465_, 0u, _e475_));
                        }
                        let _e479_ = phi_790_;
                        phi_791_ = _e479_;
                    }
                    let _e481_ = phi_791_;
                    phi_798_ = _e481_;
                } else {
                    phi_798_ = select(0f, 1f, (_e465_ == 2139095040u));
                }
                let _e483_ = phi_798_;
                let _e485_ = bitcast<u32>(_e447_.y);
                let _e486_ = (_e485_ & 2139095040u);
                if (_e486_ != 2139095040u) {
                    if (_e447_.y <= 0f) {
                        phi_831_ = 0f;
                    } else {
                        if (_e447_.y >= 1f) {
                            phi_830_ = 1f;
                        } else {
                            if (_e486_ == 2139095040u) {
                                phi_824_ = true;
                            } else {
                                phi_824_ = (_e486_ == 0u);
                            }
                            let _e495_ = phi_824_;
                            phi_830_ = bitcast<f32>(select(_e485_, 0u, _e495_));
                        }
                        let _e499_ = phi_830_;
                        phi_831_ = _e499_;
                    }
                    let _e501_ = phi_831_;
                    phi_838_ = _e501_;
                } else {
                    phi_838_ = select(0f, 1f, (_e485_ == 2139095040u));
                }
                let _e503_ = phi_838_;
                let _e505_ = bitcast<u32>(_e447_.z);
                let _e506_ = (_e505_ & 2139095040u);
                if (_e506_ != 2139095040u) {
                    if (_e447_.z <= 0f) {
                        phi_871_ = 0f;
                    } else {
                        if (_e447_.z >= 1f) {
                            phi_870_ = 1f;
                        } else {
                            if (_e506_ == 2139095040u) {
                                phi_864_ = true;
                            } else {
                                phi_864_ = (_e506_ == 0u);
                            }
                            let _e515_ = phi_864_;
                            phi_870_ = bitcast<f32>(select(_e505_, 0u, _e515_));
                        }
                        let _e519_ = phi_870_;
                        phi_871_ = _e519_;
                    }
                    let _e521_ = phi_871_;
                    phi_878_ = _e521_;
                } else {
                    phi_878_ = select(0f, 1f, (_e505_ == 2139095040u));
                }
                let _e523_ = phi_878_;
                let _e525_ = bitcast<u32>(_e447_.w);
                let _e526_ = (_e525_ & 2139095040u);
                if (_e526_ != 2139095040u) {
                    if (_e447_.w <= 0f) {
                        phi_911_ = 0f;
                    } else {
                        if (_e447_.w >= 1f) {
                            phi_910_ = 1f;
                        } else {
                            if (_e526_ == 2139095040u) {
                                phi_904_ = true;
                            } else {
                                phi_904_ = (_e526_ == 0u);
                            }
                            let _e535_ = phi_904_;
                            phi_910_ = bitcast<f32>(select(_e525_, 0u, _e535_));
                        }
                        let _e539_ = phi_910_;
                        phi_911_ = _e539_;
                    }
                    let _e541_ = phi_911_;
                    phi_918_ = _e541_;
                } else {
                    phi_918_ = select(0f, 1f, (_e525_ == 2139095040u));
                }
                let _e543_ = phi_918_;
                if _e224_ {
                    let _e548_ = (_e223_ > 0u);
                    if _e548_ {
                        let _e549_ = (_e223_ < _e216_);
                        let _e550_ = (_e157_ == 0u);
                        if _e550_ {
                            if _e549_ {
                                let _e553_ = (((_e193_ * _e216_) + _e223_) * 8u);
                                let _e554_ = (_e553_ < _e154_);
                                if _e554_ {
                                    if _e554_ {
                                    } else {
                                        break;
                                    }
                                    let _e557_ = tile_checkpoints.member[_e553_];
                                    phi_954_ = _e557_;
                                } else {
                                    phi_954_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e559_ = phi_954_;
                                let _e561_ = bitcast<u32>(_e559_.x);
                                let _e562_ = (_e561_ & 2139095040u);
                                if (_e562_ == 2139095040u) {
                                    phi_966_ = true;
                                } else {
                                    phi_966_ = (_e562_ == 0u);
                                }
                                let _e566_ = phi_966_;
                                let _e570_ = bitcast<u32>(_e559_.y);
                                let _e571_ = (_e570_ & 2139095040u);
                                if (_e571_ == 2139095040u) {
                                    phi_983_ = true;
                                } else {
                                    phi_983_ = (_e571_ == 0u);
                                }
                                let _e575_ = phi_983_;
                                let _e579_ = bitcast<u32>(_e559_.z);
                                let _e580_ = (_e579_ & 2139095040u);
                                if (_e580_ == 2139095040u) {
                                    phi_1000_ = true;
                                } else {
                                    phi_1000_ = (_e580_ == 0u);
                                }
                                let _e584_ = phi_1000_;
                                let _e588_ = bitcast<u32>(_e559_.w);
                                let _e589_ = (_e588_ & 2139095040u);
                                if (_e589_ == 2139095040u) {
                                    phi_1017_ = true;
                                } else {
                                    phi_1017_ = (_e589_ == 0u);
                                }
                                let _e593_ = phi_1017_;
                                local = array<f32, 4>(bitcast<f32>(select(_e561_, 0u, _e566_)), bitcast<f32>(select(_e570_, 0u, _e575_)), bitcast<f32>(select(_e579_, 0u, _e584_)), bitcast<f32>(select(_e588_, 0u, _e593_)));
                                let _e597_ = (_e553_ + 1u);
                                let _e598_ = (_e597_ < _e154_);
                                if _e598_ {
                                    if _e598_ {
                                    } else {
                                        break;
                                    }
                                    let _e601_ = tile_checkpoints.member[_e597_];
                                    phi_1041_ = _e601_;
                                } else {
                                    phi_1041_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e603_ = phi_1041_;
                                let _e605_ = bitcast<u32>(_e603_.x);
                                let _e606_ = (_e605_ & 2139095040u);
                                if (_e606_ == 2139095040u) {
                                    phi_1053_ = true;
                                } else {
                                    phi_1053_ = (_e606_ == 0u);
                                }
                                let _e610_ = phi_1053_;
                                let _e614_ = bitcast<u32>(_e603_.y);
                                let _e615_ = (_e614_ & 2139095040u);
                                if (_e615_ == 2139095040u) {
                                    phi_1070_ = true;
                                } else {
                                    phi_1070_ = (_e615_ == 0u);
                                }
                                let _e619_ = phi_1070_;
                                let _e623_ = bitcast<u32>(_e603_.z);
                                let _e624_ = (_e623_ & 2139095040u);
                                if (_e624_ == 2139095040u) {
                                    phi_1087_ = true;
                                } else {
                                    phi_1087_ = (_e624_ == 0u);
                                }
                                let _e628_ = phi_1087_;
                                let _e632_ = bitcast<u32>(_e603_.w);
                                let _e633_ = (_e632_ & 2139095040u);
                                if (_e633_ == 2139095040u) {
                                    phi_1104_ = true;
                                } else {
                                    phi_1104_ = (_e633_ == 0u);
                                }
                                let _e637_ = phi_1104_;
                                local_1_ = array<f32, 4>(bitcast<f32>(select(_e605_, 0u, _e610_)), bitcast<f32>(select(_e614_, 0u, _e619_)), bitcast<f32>(select(_e623_, 0u, _e628_)), bitcast<f32>(select(_e632_, 0u, _e637_)));
                                let _e641_ = (_e553_ + 2u);
                                let _e642_ = (_e641_ < _e154_);
                                if _e642_ {
                                    if _e642_ {
                                    } else {
                                        break;
                                    }
                                    let _e645_ = tile_checkpoints.member[_e641_];
                                    phi_1128_ = _e645_;
                                } else {
                                    phi_1128_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e647_ = phi_1128_;
                                let _e649_ = bitcast<u32>(_e647_.x);
                                let _e650_ = (_e649_ & 2139095040u);
                                if (_e650_ == 2139095040u) {
                                    phi_1140_ = true;
                                } else {
                                    phi_1140_ = (_e650_ == 0u);
                                }
                                let _e654_ = phi_1140_;
                                let _e658_ = bitcast<u32>(_e647_.y);
                                let _e659_ = (_e658_ & 2139095040u);
                                if (_e659_ == 2139095040u) {
                                    phi_1157_ = true;
                                } else {
                                    phi_1157_ = (_e659_ == 0u);
                                }
                                let _e663_ = phi_1157_;
                                let _e667_ = bitcast<u32>(_e647_.z);
                                let _e668_ = (_e667_ & 2139095040u);
                                if (_e668_ == 2139095040u) {
                                    phi_1174_ = true;
                                } else {
                                    phi_1174_ = (_e668_ == 0u);
                                }
                                let _e672_ = phi_1174_;
                                let _e676_ = bitcast<u32>(_e647_.w);
                                let _e677_ = (_e676_ & 2139095040u);
                                if (_e677_ == 2139095040u) {
                                    phi_1191_ = true;
                                } else {
                                    phi_1191_ = (_e677_ == 0u);
                                }
                                let _e681_ = phi_1191_;
                                local_2_ = array<f32, 4>(bitcast<f32>(select(_e649_, 0u, _e654_)), bitcast<f32>(select(_e658_, 0u, _e663_)), bitcast<f32>(select(_e667_, 0u, _e672_)), bitcast<f32>(select(_e676_, 0u, _e681_)));
                                let _e685_ = (_e553_ + 3u);
                                let _e686_ = (_e685_ < _e154_);
                                if _e686_ {
                                    if _e686_ {
                                    } else {
                                        break;
                                    }
                                    let _e689_ = tile_checkpoints.member[_e685_];
                                    phi_1215_ = _e689_;
                                } else {
                                    phi_1215_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e691_ = phi_1215_;
                                let _e693_ = bitcast<u32>(_e691_.x);
                                let _e694_ = (_e693_ & 2139095040u);
                                if (_e694_ == 2139095040u) {
                                    phi_1227_ = true;
                                } else {
                                    phi_1227_ = (_e694_ == 0u);
                                }
                                let _e698_ = phi_1227_;
                                let _e702_ = bitcast<u32>(_e691_.y);
                                let _e703_ = (_e702_ & 2139095040u);
                                if (_e703_ == 2139095040u) {
                                    phi_1244_ = true;
                                } else {
                                    phi_1244_ = (_e703_ == 0u);
                                }
                                let _e707_ = phi_1244_;
                                let _e711_ = bitcast<u32>(_e691_.z);
                                let _e712_ = (_e711_ & 2139095040u);
                                if (_e712_ == 2139095040u) {
                                    phi_1261_ = true;
                                } else {
                                    phi_1261_ = (_e712_ == 0u);
                                }
                                let _e716_ = phi_1261_;
                                let _e720_ = bitcast<u32>(_e691_.w);
                                let _e721_ = (_e720_ & 2139095040u);
                                if (_e721_ == 2139095040u) {
                                    phi_1278_ = true;
                                } else {
                                    phi_1278_ = (_e721_ == 0u);
                                }
                                let _e725_ = phi_1278_;
                                local_3_ = array<f32, 4>(bitcast<f32>(select(_e693_, 0u, _e698_)), bitcast<f32>(select(_e702_, 0u, _e707_)), bitcast<f32>(select(_e711_, 0u, _e716_)), bitcast<f32>(select(_e720_, 0u, _e725_)));
                                let _e729_ = (_e553_ + 4u);
                                let _e730_ = (_e729_ < _e154_);
                                if _e730_ {
                                    if _e730_ {
                                    } else {
                                        break;
                                    }
                                    let _e733_ = tile_checkpoints.member[_e729_];
                                    phi_1302_ = _e733_;
                                } else {
                                    phi_1302_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e735_ = phi_1302_;
                                let _e737_ = bitcast<u32>(_e735_.x);
                                let _e738_ = (_e737_ & 2139095040u);
                                if (_e738_ == 2139095040u) {
                                    phi_1314_ = true;
                                } else {
                                    phi_1314_ = (_e738_ == 0u);
                                }
                                let _e742_ = phi_1314_;
                                let _e746_ = bitcast<u32>(_e735_.y);
                                let _e747_ = (_e746_ & 2139095040u);
                                if (_e747_ == 2139095040u) {
                                    phi_1331_ = true;
                                } else {
                                    phi_1331_ = (_e747_ == 0u);
                                }
                                let _e751_ = phi_1331_;
                                let _e755_ = bitcast<u32>(_e735_.z);
                                let _e756_ = (_e755_ & 2139095040u);
                                if (_e756_ == 2139095040u) {
                                    phi_1348_ = true;
                                } else {
                                    phi_1348_ = (_e756_ == 0u);
                                }
                                let _e760_ = phi_1348_;
                                let _e764_ = bitcast<u32>(_e735_.w);
                                let _e765_ = (_e764_ & 2139095040u);
                                if (_e765_ == 2139095040u) {
                                    phi_1365_ = true;
                                } else {
                                    phi_1365_ = (_e765_ == 0u);
                                }
                                let _e769_ = phi_1365_;
                                local_4_ = array<f32, 4>(bitcast<f32>(select(_e737_, 0u, _e742_)), bitcast<f32>(select(_e746_, 0u, _e751_)), bitcast<f32>(select(_e755_, 0u, _e760_)), bitcast<f32>(select(_e764_, 0u, _e769_)));
                                let _e773_ = (_e553_ + 5u);
                                let _e774_ = (_e773_ < _e154_);
                                if _e774_ {
                                    if _e774_ {
                                    } else {
                                        break;
                                    }
                                    let _e777_ = tile_checkpoints.member[_e773_];
                                    phi_1389_ = _e777_;
                                } else {
                                    phi_1389_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e779_ = phi_1389_;
                                let _e781_ = bitcast<u32>(_e779_.x);
                                let _e782_ = (_e781_ & 2139095040u);
                                if (_e782_ == 2139095040u) {
                                    phi_1401_ = true;
                                } else {
                                    phi_1401_ = (_e782_ == 0u);
                                }
                                let _e786_ = phi_1401_;
                                let _e790_ = bitcast<u32>(_e779_.y);
                                let _e791_ = (_e790_ & 2139095040u);
                                if (_e791_ == 2139095040u) {
                                    phi_1418_ = true;
                                } else {
                                    phi_1418_ = (_e791_ == 0u);
                                }
                                let _e795_ = phi_1418_;
                                let _e799_ = bitcast<u32>(_e779_.z);
                                let _e800_ = (_e799_ & 2139095040u);
                                if (_e800_ == 2139095040u) {
                                    phi_1435_ = true;
                                } else {
                                    phi_1435_ = (_e800_ == 0u);
                                }
                                let _e804_ = phi_1435_;
                                let _e808_ = bitcast<u32>(_e779_.w);
                                let _e809_ = (_e808_ & 2139095040u);
                                if (_e809_ == 2139095040u) {
                                    phi_1452_ = true;
                                } else {
                                    phi_1452_ = (_e809_ == 0u);
                                }
                                let _e813_ = phi_1452_;
                                local_5_ = array<f32, 4>(bitcast<f32>(select(_e781_, 0u, _e786_)), bitcast<f32>(select(_e790_, 0u, _e795_)), bitcast<f32>(select(_e799_, 0u, _e804_)), bitcast<f32>(select(_e808_, 0u, _e813_)));
                                let _e817_ = (_e553_ + 6u);
                                let _e818_ = (_e817_ < _e154_);
                                if _e818_ {
                                    if _e818_ {
                                    } else {
                                        break;
                                    }
                                    let _e821_ = tile_checkpoints.member[_e817_];
                                    phi_1476_ = _e821_;
                                } else {
                                    phi_1476_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e823_ = phi_1476_;
                                let _e825_ = bitcast<u32>(_e823_.x);
                                let _e826_ = (_e825_ & 2139095040u);
                                if (_e826_ == 2139095040u) {
                                    phi_1488_ = true;
                                } else {
                                    phi_1488_ = (_e826_ == 0u);
                                }
                                let _e830_ = phi_1488_;
                                let _e832_ = bitcast<f32>(select(_e825_, 0u, _e830_));
                                if (_e832_ < -1000000000000000000f) {
                                    phi_1505_ = -1000000000000000000f;
                                } else {
                                    phi_1505_ = select(_e832_, 1000000000000000000f, (_e832_ > 1000000000000000000f));
                                }
                                let _e837_ = phi_1505_;
                                let _e839_ = bitcast<u32>(_e823_.y);
                                let _e840_ = (_e839_ & 2139095040u);
                                if (_e840_ == 2139095040u) {
                                    phi_1517_ = true;
                                } else {
                                    phi_1517_ = (_e840_ == 0u);
                                }
                                let _e844_ = phi_1517_;
                                let _e846_ = bitcast<f32>(select(_e839_, 0u, _e844_));
                                if (_e846_ < -1000000000000000000f) {
                                    phi_1534_ = -1000000000000000000f;
                                } else {
                                    phi_1534_ = select(_e846_, 1000000000000000000f, (_e846_ > 1000000000000000000f));
                                }
                                let _e851_ = phi_1534_;
                                let _e853_ = bitcast<u32>(_e823_.z);
                                let _e854_ = (_e853_ & 2139095040u);
                                if (_e854_ == 2139095040u) {
                                    phi_1546_ = true;
                                } else {
                                    phi_1546_ = (_e854_ == 0u);
                                }
                                let _e858_ = phi_1546_;
                                let _e860_ = bitcast<f32>(select(_e853_, 0u, _e858_));
                                if (_e860_ < -1000000000000000000f) {
                                    phi_1563_ = -1000000000000000000f;
                                } else {
                                    phi_1563_ = select(_e860_, 1000000000000000000f, (_e860_ > 1000000000000000000f));
                                }
                                let _e865_ = phi_1563_;
                                let _e867_ = bitcast<u32>(_e823_.w);
                                let _e868_ = (_e867_ & 2139095040u);
                                if (_e868_ == 2139095040u) {
                                    phi_1575_ = true;
                                } else {
                                    phi_1575_ = (_e868_ == 0u);
                                }
                                let _e872_ = phi_1575_;
                                let _e874_ = bitcast<f32>(select(_e867_, 0u, _e872_));
                                if (_e874_ < -1000000000000000000f) {
                                    phi_1592_ = -1000000000000000000f;
                                } else {
                                    phi_1592_ = select(_e874_, 1000000000000000000f, (_e874_ > 1000000000000000000f));
                                }
                                let _e879_ = phi_1592_;
                                local_6_ = array<f32, 4>(_e837_, _e851_, _e865_, _e879_);
                                let _e881_ = (_e553_ + 7u);
                                let _e882_ = (_e881_ < _e154_);
                                if _e882_ {
                                    if _e882_ {
                                    } else {
                                        break;
                                    }
                                    let _e885_ = tile_checkpoints.member[_e881_];
                                    phi_1611_ = _e885_;
                                } else {
                                    phi_1611_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e887_ = phi_1611_;
                                let _e889_ = bitcast<u32>(_e887_.x);
                                let _e890_ = (_e889_ & 2139095040u);
                                if (_e890_ == 2139095040u) {
                                    phi_1623_ = true;
                                } else {
                                    phi_1623_ = (_e890_ == 0u);
                                }
                                let _e894_ = phi_1623_;
                                let _e896_ = bitcast<f32>(select(_e889_, 0u, _e894_));
                                if (_e896_ < -1000000000000000000f) {
                                    phi_1640_ = -1000000000000000000f;
                                } else {
                                    phi_1640_ = select(_e896_, 1000000000000000000f, (_e896_ > 1000000000000000000f));
                                }
                                let _e901_ = phi_1640_;
                                let _e903_ = bitcast<u32>(_e887_.y);
                                let _e904_ = (_e903_ & 2139095040u);
                                if (_e904_ == 2139095040u) {
                                    phi_1652_ = true;
                                } else {
                                    phi_1652_ = (_e904_ == 0u);
                                }
                                let _e908_ = phi_1652_;
                                let _e910_ = bitcast<f32>(select(_e903_, 0u, _e908_));
                                if (_e910_ < -1000000000000000000f) {
                                    phi_1669_ = -1000000000000000000f;
                                } else {
                                    phi_1669_ = select(_e910_, 1000000000000000000f, (_e910_ > 1000000000000000000f));
                                }
                                let _e915_ = phi_1669_;
                                let _e917_ = bitcast<u32>(_e887_.z);
                                let _e918_ = (_e917_ & 2139095040u);
                                if (_e918_ == 2139095040u) {
                                    phi_1681_ = true;
                                } else {
                                    phi_1681_ = (_e918_ == 0u);
                                }
                                let _e922_ = phi_1681_;
                                let _e924_ = bitcast<f32>(select(_e917_, 0u, _e922_));
                                if (_e924_ < -1000000000000000000f) {
                                    phi_1698_ = -1000000000000000000f;
                                } else {
                                    phi_1698_ = select(_e924_, 1000000000000000000f, (_e924_ > 1000000000000000000f));
                                }
                                let _e929_ = phi_1698_;
                                let _e931_ = bitcast<u32>(_e887_.w);
                                let _e932_ = (_e931_ & 2139095040u);
                                if (_e932_ == 2139095040u) {
                                    phi_1710_ = true;
                                } else {
                                    phi_1710_ = (_e932_ == 0u);
                                }
                                let _e936_ = phi_1710_;
                                let _e938_ = bitcast<f32>(select(_e931_, 0u, _e936_));
                                if (_e938_ < -1000000000000000000f) {
                                    phi_1727_ = -1000000000000000000f;
                                } else {
                                    phi_1727_ = select(_e938_, 1000000000000000000f, (_e938_ > 1000000000000000000f));
                                }
                                let _e943_ = phi_1727_;
                                local_7_ = array<f32, 4>(_e901_, _e915_, _e929_, _e943_);
                                phi_1736_ = type_15_(0u, 4u);
                                loop {
                                    let _e946_ = phi_1736_;
                                    if (_e946_.member < _e946_.member_1_) {
                                        phi_1737_ = type_15_((_e946_.member + 1u), _e946_.member_1_);
                                        phi_1760_ = type_15_(1u, _e946_.member);
                                    } else {
                                        phi_1737_ = _e946_;
                                        phi_1760_ = type_15_(0u, u32());
                                    }
                                    let _e959_ = phi_1737_;
                                    let _e961_ = phi_1760_;
                                    let _e965_ = (bitcast<i32>(_e961_.member) != 0i);
                                    if _e965_ {
                                        let _e966_ = (_e961_.member_1_ < 4u);
                                        if _e966_ {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        let _e968_ = local[_e961_.member_1_];
                                        if (_e961_.member_1_ < 256u) {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        aux_scratch[_e961_.member_1_] = _e968_;
                                        if _e966_ {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        let _e972_ = local_2_[_e961_.member_1_];
                                        let _e973_ = (4u + _e961_.member_1_);
                                        if (_e973_ < 256u) {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        aux_scratch[_e973_] = _e972_;
                                        if _e966_ {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        let _e977_ = local_1_[_e961_.member_1_];
                                        let _e978_ = (8u + _e961_.member_1_);
                                        if (_e978_ < 256u) {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        aux_scratch[_e978_] = _e977_;
                                        if _e966_ {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        let _e982_ = local_3_[_e961_.member_1_];
                                        let _e983_ = (12u + _e961_.member_1_);
                                        if (_e983_ < 256u) {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        aux_scratch[_e983_] = _e982_;
                                        if _e966_ {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        let _e987_ = local_4_[_e961_.member_1_];
                                        let _e988_ = (16u + _e961_.member_1_);
                                        if (_e988_ < 256u) {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        aux_scratch[_e988_] = _e987_;
                                        if _e966_ {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        let _e992_ = local_5_[_e961_.member_1_];
                                        let _e993_ = (20u + _e961_.member_1_);
                                        if (_e993_ < 256u) {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        aux_scratch[_e993_] = _e992_;
                                        if _e966_ {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        let _e997_ = local_6_[_e961_.member_1_];
                                        let _e998_ = (24u + _e961_.member_1_);
                                        if (_e998_ < 256u) {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        aux_scratch[_e998_] = _e997_;
                                        if _e966_ {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        let _e1002_ = local_7_[_e961_.member_1_];
                                        let _e1003_ = (28u + _e961_.member_1_);
                                        if (_e1003_ < 256u) {
                                        } else {
                                            phi_34691_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1003_] = _e1002_;
                                    }
                                    phi_1736_ = _e959_;
                                    phi_34691_ = false;
                                    if !(_e965_) {
                                        break;
                                    }
                                }
                                let _e1008_ = phi_34691_;
                                if _e1008_ {
                                    break;
                                }
                                phi_34863_ = _e1008_;
                            } else {
                                phi_34863_ = false;
                            }
                            let _e1010_ = phi_34863_;
                            phi_34862_ = _e1010_;
                            phi_1880_ = select(true, false, _e549_);
                        } else {
                            phi_34862_ = false;
                            phi_1880_ = false;
                        }
                        let _e1013_ = phi_34862_;
                        let _e1015_ = phi_1880_;
                        phi_34861_ = _e1013_;
                        phi_1882_ = _e549_;
                        phi_1886_ = _e1015_;
                        phi_1887_ = select(false, _e549_, _e550_);
                    } else {
                        phi_34861_ = false;
                        phi_1882_ = bool();
                        phi_1886_ = false;
                        phi_1887_ = false;
                    }
                    let _e1018_ = phi_34861_;
                    let _e1020_ = phi_1882_;
                    let _e1022_ = phi_1886_;
                    let _e1024_ = phi_1887_;
                    phi_34860_ = _e1018_;
                    phi_1888_ = _e1020_;
                    phi_1891_ = select(true, false, _e548_);
                    phi_1892_ = _e1022_;
                    phi_1893_ = _e1024_;
                } else {
                    phi_34860_ = false;
                    phi_1888_ = bool();
                    phi_1891_ = true;
                    phi_1892_ = false;
                    phi_1893_ = false;
                }
                let _e1027_ = phi_34860_;
                let _e1029_ = phi_1888_;
                let _e1031_ = phi_1891_;
                let _e1033_ = phi_1892_;
                let _e1035_ = phi_1893_;
                if _e1031_ {
                    phi_1903_ = (_e157_ == 0u);
                } else {
                    phi_1903_ = _e1033_;
                }
                let _e1038_ = phi_1903_;
                let _e1039_ = select(_e1029_, false, _e1031_);
                if _e1038_ {
                    phi_1910_ = type_15_(0u, 32u);
                    loop {
                        let _e1042_ = phi_1910_;
                        if (_e1042_.member < _e1042_.member_1_) {
                            phi_1911_ = type_15_((_e1042_.member + 1u), _e1042_.member_1_);
                            phi_1934_ = type_15_(1u, _e1042_.member);
                        } else {
                            phi_1911_ = _e1042_;
                            phi_1934_ = type_15_(0u, u32());
                        }
                        let _e1055_ = phi_1911_;
                        let _e1057_ = phi_1934_;
                        let _e1061_ = (bitcast<i32>(_e1057_.member) != 0i);
                        if _e1061_ {
                            if (_e1057_.member_1_ < 256u) {
                            } else {
                                phi_34855_ = true;
                                break;
                            }
                            aux_scratch[_e1057_.member_1_] = 0f;
                        }
                        phi_1910_ = _e1055_;
                        phi_34855_ = _e1027_;
                        if !(_e1061_) {
                            break;
                        }
                    }
                    let _e1066_ = phi_34855_;
                    if _e1066_ {
                        break;
                    }
                    phi_34894_ = _e1066_;
                } else {
                    phi_34894_ = _e1027_;
                }
                let _e1068_ = phi_34894_;
                workgroupBarrier();
                phi_34891_ = _e1068_;
                phi_1960_ = type_15_(_e291_, _e293_);
                loop {
                    let _e1075_ = phi_34891_;
                    let _e1077_ = phi_1960_;
                    if (_e1077_.member < _e1077_.member_1_) {
                        phi_1961_ = type_15_((_e1077_.member + 1u), _e1077_.member_1_);
                        phi_1984_ = type_15_(1u, _e1077_.member);
                    } else {
                        phi_1961_ = _e1077_;
                        phi_1984_ = type_15_(0u, u32());
                    }
                    let _e1090_ = phi_1961_;
                    let _e1092_ = phi_1984_;
                    let _e1096_ = (bitcast<i32>(_e1092_.member) != 0i);
                    if _e1096_ {
                        let _e1098_ = (_e157_ == 0u);
                        if _e1098_ {
                            if select(select(_e1029_, _e1039_, _e1038_), _e1039_, select(true, false, select(select(_e1035_, false, _e1031_), true, _e1038_))) {
                                phi_2005_ = select(true, false, (_e1092_.member_1_ == _e223_));
                            } else {
                                phi_2005_ = true;
                            }
                            let _e1102_ = phi_2005_;
                            if _e1102_ {
                                let _e1105_ = (((_e193_ * _e216_) + _e1092_.member_1_) * 8u);
                                let _e1107_ = aux_scratch[0];
                                let _e1109_ = aux_scratch[1];
                                let _e1111_ = aux_scratch[2];
                                let _e1113_ = aux_scratch[3];
                                let _e1115_ = aux_scratch[4];
                                let _e1117_ = aux_scratch[5];
                                let _e1119_ = aux_scratch[6];
                                let _e1121_ = aux_scratch[7];
                                let _e1123_ = aux_scratch[8];
                                let _e1125_ = aux_scratch[9];
                                let _e1127_ = aux_scratch[10];
                                let _e1129_ = aux_scratch[11];
                                let _e1131_ = aux_scratch[12];
                                let _e1133_ = aux_scratch[13];
                                let _e1135_ = aux_scratch[14];
                                let _e1137_ = aux_scratch[15];
                                let _e1139_ = aux_scratch[16];
                                let _e1141_ = aux_scratch[17];
                                let _e1143_ = aux_scratch[18];
                                let _e1145_ = aux_scratch[19];
                                let _e1147_ = aux_scratch[20];
                                let _e1149_ = aux_scratch[21];
                                let _e1151_ = aux_scratch[22];
                                let _e1153_ = aux_scratch[23];
                                let _e1155_ = aux_scratch[24];
                                let _e1157_ = aux_scratch[25];
                                let _e1159_ = aux_scratch[26];
                                let _e1161_ = aux_scratch[27];
                                let _e1163_ = aux_scratch[28];
                                let _e1165_ = aux_scratch[29];
                                let _e1167_ = aux_scratch[30];
                                let _e1169_ = aux_scratch[31];
                                let _e1170_ = bitcast<u32>(_e1107_);
                                let _e1171_ = (_e1170_ & 2139095040u);
                                if (_e1171_ == 2139095040u) {
                                    phi_2086_ = true;
                                } else {
                                    phi_2086_ = (_e1171_ == 0u);
                                }
                                let _e1175_ = phi_2086_;
                                let _e1178_ = bitcast<u32>(_e1109_);
                                let _e1179_ = (_e1178_ & 2139095040u);
                                if (_e1179_ == 2139095040u) {
                                    phi_2102_ = true;
                                } else {
                                    phi_2102_ = (_e1179_ == 0u);
                                }
                                let _e1183_ = phi_2102_;
                                let _e1186_ = bitcast<u32>(_e1111_);
                                let _e1187_ = (_e1186_ & 2139095040u);
                                if (_e1187_ == 2139095040u) {
                                    phi_2118_ = true;
                                } else {
                                    phi_2118_ = (_e1187_ == 0u);
                                }
                                let _e1191_ = phi_2118_;
                                let _e1194_ = bitcast<u32>(_e1113_);
                                let _e1195_ = (_e1194_ & 2139095040u);
                                if (_e1195_ == 2139095040u) {
                                    phi_2134_ = true;
                                } else {
                                    phi_2134_ = (_e1195_ == 0u);
                                }
                                let _e1199_ = phi_2134_;
                                let _e1203_ = (_e1105_ < _e154_);
                                if _e1203_ {
                                    if _e1203_ {
                                    } else {
                                        phi_37458_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1105_] = vec4<f32>(bitcast<f32>(select(_e1170_, 0u, _e1175_)), bitcast<f32>(select(_e1178_, 0u, _e1183_)), bitcast<f32>(select(_e1186_, 0u, _e1191_)), bitcast<f32>(select(_e1194_, 0u, _e1199_)));
                                }
                                let _e1206_ = (_e1105_ + 1u);
                                let _e1207_ = bitcast<u32>(_e1123_);
                                let _e1208_ = (_e1207_ & 2139095040u);
                                if (_e1208_ == 2139095040u) {
                                    phi_2162_ = true;
                                } else {
                                    phi_2162_ = (_e1208_ == 0u);
                                }
                                let _e1212_ = phi_2162_;
                                let _e1215_ = bitcast<u32>(_e1125_);
                                let _e1216_ = (_e1215_ & 2139095040u);
                                if (_e1216_ == 2139095040u) {
                                    phi_2178_ = true;
                                } else {
                                    phi_2178_ = (_e1216_ == 0u);
                                }
                                let _e1220_ = phi_2178_;
                                let _e1223_ = bitcast<u32>(_e1127_);
                                let _e1224_ = (_e1223_ & 2139095040u);
                                if (_e1224_ == 2139095040u) {
                                    phi_2194_ = true;
                                } else {
                                    phi_2194_ = (_e1224_ == 0u);
                                }
                                let _e1228_ = phi_2194_;
                                let _e1231_ = bitcast<u32>(_e1129_);
                                let _e1232_ = (_e1231_ & 2139095040u);
                                if (_e1232_ == 2139095040u) {
                                    phi_2210_ = true;
                                } else {
                                    phi_2210_ = (_e1232_ == 0u);
                                }
                                let _e1236_ = phi_2210_;
                                let _e1240_ = (_e1206_ < _e154_);
                                if _e1240_ {
                                    if _e1240_ {
                                    } else {
                                        phi_37458_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1206_] = vec4<f32>(bitcast<f32>(select(_e1207_, 0u, _e1212_)), bitcast<f32>(select(_e1215_, 0u, _e1220_)), bitcast<f32>(select(_e1223_, 0u, _e1228_)), bitcast<f32>(select(_e1231_, 0u, _e1236_)));
                                }
                                let _e1243_ = (_e1105_ + 2u);
                                let _e1244_ = bitcast<u32>(_e1115_);
                                let _e1245_ = (_e1244_ & 2139095040u);
                                if (_e1245_ == 2139095040u) {
                                    phi_2238_ = true;
                                } else {
                                    phi_2238_ = (_e1245_ == 0u);
                                }
                                let _e1249_ = phi_2238_;
                                let _e1252_ = bitcast<u32>(_e1117_);
                                let _e1253_ = (_e1252_ & 2139095040u);
                                if (_e1253_ == 2139095040u) {
                                    phi_2254_ = true;
                                } else {
                                    phi_2254_ = (_e1253_ == 0u);
                                }
                                let _e1257_ = phi_2254_;
                                let _e1260_ = bitcast<u32>(_e1119_);
                                let _e1261_ = (_e1260_ & 2139095040u);
                                if (_e1261_ == 2139095040u) {
                                    phi_2270_ = true;
                                } else {
                                    phi_2270_ = (_e1261_ == 0u);
                                }
                                let _e1265_ = phi_2270_;
                                let _e1268_ = bitcast<u32>(_e1121_);
                                let _e1269_ = (_e1268_ & 2139095040u);
                                if (_e1269_ == 2139095040u) {
                                    phi_2286_ = true;
                                } else {
                                    phi_2286_ = (_e1269_ == 0u);
                                }
                                let _e1273_ = phi_2286_;
                                let _e1277_ = (_e1243_ < _e154_);
                                if _e1277_ {
                                    if _e1277_ {
                                    } else {
                                        phi_37458_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1243_] = vec4<f32>(bitcast<f32>(select(_e1244_, 0u, _e1249_)), bitcast<f32>(select(_e1252_, 0u, _e1257_)), bitcast<f32>(select(_e1260_, 0u, _e1265_)), bitcast<f32>(select(_e1268_, 0u, _e1273_)));
                                }
                                let _e1280_ = (_e1105_ + 3u);
                                let _e1281_ = bitcast<u32>(_e1131_);
                                let _e1282_ = (_e1281_ & 2139095040u);
                                if (_e1282_ == 2139095040u) {
                                    phi_2314_ = true;
                                } else {
                                    phi_2314_ = (_e1282_ == 0u);
                                }
                                let _e1286_ = phi_2314_;
                                let _e1289_ = bitcast<u32>(_e1133_);
                                let _e1290_ = (_e1289_ & 2139095040u);
                                if (_e1290_ == 2139095040u) {
                                    phi_2330_ = true;
                                } else {
                                    phi_2330_ = (_e1290_ == 0u);
                                }
                                let _e1294_ = phi_2330_;
                                let _e1297_ = bitcast<u32>(_e1135_);
                                let _e1298_ = (_e1297_ & 2139095040u);
                                if (_e1298_ == 2139095040u) {
                                    phi_2346_ = true;
                                } else {
                                    phi_2346_ = (_e1298_ == 0u);
                                }
                                let _e1302_ = phi_2346_;
                                let _e1305_ = bitcast<u32>(_e1137_);
                                let _e1306_ = (_e1305_ & 2139095040u);
                                if (_e1306_ == 2139095040u) {
                                    phi_2362_ = true;
                                } else {
                                    phi_2362_ = (_e1306_ == 0u);
                                }
                                let _e1310_ = phi_2362_;
                                let _e1314_ = (_e1280_ < _e154_);
                                if _e1314_ {
                                    if _e1314_ {
                                    } else {
                                        phi_37458_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1280_] = vec4<f32>(bitcast<f32>(select(_e1281_, 0u, _e1286_)), bitcast<f32>(select(_e1289_, 0u, _e1294_)), bitcast<f32>(select(_e1297_, 0u, _e1302_)), bitcast<f32>(select(_e1305_, 0u, _e1310_)));
                                }
                                let _e1317_ = (_e1105_ + 4u);
                                let _e1318_ = bitcast<u32>(_e1139_);
                                let _e1319_ = (_e1318_ & 2139095040u);
                                if (_e1319_ == 2139095040u) {
                                    phi_2390_ = true;
                                } else {
                                    phi_2390_ = (_e1319_ == 0u);
                                }
                                let _e1323_ = phi_2390_;
                                let _e1326_ = bitcast<u32>(_e1141_);
                                let _e1327_ = (_e1326_ & 2139095040u);
                                if (_e1327_ == 2139095040u) {
                                    phi_2406_ = true;
                                } else {
                                    phi_2406_ = (_e1327_ == 0u);
                                }
                                let _e1331_ = phi_2406_;
                                let _e1334_ = bitcast<u32>(_e1143_);
                                let _e1335_ = (_e1334_ & 2139095040u);
                                if (_e1335_ == 2139095040u) {
                                    phi_2422_ = true;
                                } else {
                                    phi_2422_ = (_e1335_ == 0u);
                                }
                                let _e1339_ = phi_2422_;
                                let _e1342_ = bitcast<u32>(_e1145_);
                                let _e1343_ = (_e1342_ & 2139095040u);
                                if (_e1343_ == 2139095040u) {
                                    phi_2438_ = true;
                                } else {
                                    phi_2438_ = (_e1343_ == 0u);
                                }
                                let _e1347_ = phi_2438_;
                                let _e1351_ = (_e1317_ < _e154_);
                                if _e1351_ {
                                    if _e1351_ {
                                    } else {
                                        phi_37458_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1317_] = vec4<f32>(bitcast<f32>(select(_e1318_, 0u, _e1323_)), bitcast<f32>(select(_e1326_, 0u, _e1331_)), bitcast<f32>(select(_e1334_, 0u, _e1339_)), bitcast<f32>(select(_e1342_, 0u, _e1347_)));
                                }
                                let _e1354_ = (_e1105_ + 5u);
                                let _e1355_ = bitcast<u32>(_e1147_);
                                let _e1356_ = (_e1355_ & 2139095040u);
                                if (_e1356_ == 2139095040u) {
                                    phi_2466_ = true;
                                } else {
                                    phi_2466_ = (_e1356_ == 0u);
                                }
                                let _e1360_ = phi_2466_;
                                let _e1363_ = bitcast<u32>(_e1149_);
                                let _e1364_ = (_e1363_ & 2139095040u);
                                if (_e1364_ == 2139095040u) {
                                    phi_2482_ = true;
                                } else {
                                    phi_2482_ = (_e1364_ == 0u);
                                }
                                let _e1368_ = phi_2482_;
                                let _e1371_ = bitcast<u32>(_e1151_);
                                let _e1372_ = (_e1371_ & 2139095040u);
                                if (_e1372_ == 2139095040u) {
                                    phi_2498_ = true;
                                } else {
                                    phi_2498_ = (_e1372_ == 0u);
                                }
                                let _e1376_ = phi_2498_;
                                let _e1379_ = bitcast<u32>(_e1153_);
                                let _e1380_ = (_e1379_ & 2139095040u);
                                if (_e1380_ == 2139095040u) {
                                    phi_2514_ = true;
                                } else {
                                    phi_2514_ = (_e1380_ == 0u);
                                }
                                let _e1384_ = phi_2514_;
                                let _e1388_ = (_e1354_ < _e154_);
                                if _e1388_ {
                                    if _e1388_ {
                                    } else {
                                        phi_37458_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1354_] = vec4<f32>(bitcast<f32>(select(_e1355_, 0u, _e1360_)), bitcast<f32>(select(_e1363_, 0u, _e1368_)), bitcast<f32>(select(_e1371_, 0u, _e1376_)), bitcast<f32>(select(_e1379_, 0u, _e1384_)));
                                }
                                let _e1391_ = (_e1105_ + 6u);
                                let _e1392_ = bitcast<u32>(_e1155_);
                                let _e1393_ = (_e1392_ & 2139095040u);
                                if (_e1393_ == 2139095040u) {
                                    phi_2542_ = true;
                                } else {
                                    phi_2542_ = (_e1393_ == 0u);
                                }
                                let _e1397_ = phi_2542_;
                                let _e1400_ = bitcast<u32>(_e1157_);
                                let _e1401_ = (_e1400_ & 2139095040u);
                                if (_e1401_ == 2139095040u) {
                                    phi_2558_ = true;
                                } else {
                                    phi_2558_ = (_e1401_ == 0u);
                                }
                                let _e1405_ = phi_2558_;
                                let _e1408_ = bitcast<u32>(_e1159_);
                                let _e1409_ = (_e1408_ & 2139095040u);
                                if (_e1409_ == 2139095040u) {
                                    phi_2574_ = true;
                                } else {
                                    phi_2574_ = (_e1409_ == 0u);
                                }
                                let _e1413_ = phi_2574_;
                                let _e1416_ = bitcast<u32>(_e1161_);
                                let _e1417_ = (_e1416_ & 2139095040u);
                                if (_e1417_ == 2139095040u) {
                                    phi_2590_ = true;
                                } else {
                                    phi_2590_ = (_e1417_ == 0u);
                                }
                                let _e1421_ = phi_2590_;
                                let _e1425_ = (_e1391_ < _e154_);
                                if _e1425_ {
                                    if _e1425_ {
                                    } else {
                                        phi_37458_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1391_] = vec4<f32>(bitcast<f32>(select(_e1392_, 0u, _e1397_)), bitcast<f32>(select(_e1400_, 0u, _e1405_)), bitcast<f32>(select(_e1408_, 0u, _e1413_)), bitcast<f32>(select(_e1416_, 0u, _e1421_)));
                                }
                                let _e1428_ = (_e1105_ + 7u);
                                let _e1429_ = bitcast<u32>(_e1163_);
                                let _e1430_ = (_e1429_ & 2139095040u);
                                if (_e1430_ == 2139095040u) {
                                    phi_2618_ = true;
                                } else {
                                    phi_2618_ = (_e1430_ == 0u);
                                }
                                let _e1434_ = phi_2618_;
                                let _e1437_ = bitcast<u32>(_e1165_);
                                let _e1438_ = (_e1437_ & 2139095040u);
                                if (_e1438_ == 2139095040u) {
                                    phi_2634_ = true;
                                } else {
                                    phi_2634_ = (_e1438_ == 0u);
                                }
                                let _e1442_ = phi_2634_;
                                let _e1445_ = bitcast<u32>(_e1167_);
                                let _e1446_ = (_e1445_ & 2139095040u);
                                if (_e1446_ == 2139095040u) {
                                    phi_2650_ = true;
                                } else {
                                    phi_2650_ = (_e1446_ == 0u);
                                }
                                let _e1450_ = phi_2650_;
                                let _e1453_ = bitcast<u32>(_e1169_);
                                let _e1454_ = (_e1453_ & 2139095040u);
                                if (_e1454_ == 2139095040u) {
                                    phi_2666_ = true;
                                } else {
                                    phi_2666_ = (_e1454_ == 0u);
                                }
                                let _e1458_ = phi_2666_;
                                let _e1462_ = (_e1428_ < _e154_);
                                if _e1462_ {
                                    if _e1462_ {
                                    } else {
                                        phi_37458_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1428_] = vec4<f32>(bitcast<f32>(select(_e1429_, 0u, _e1434_)), bitcast<f32>(select(_e1437_, 0u, _e1442_)), bitcast<f32>(select(_e1445_, 0u, _e1450_)), bitcast<f32>(select(_e1453_, 0u, _e1458_)));
                                }
                            }
                        }
                        workgroupBarrier();
                        phi_34888_ = _e1075_;
                        phi_2685_ = type_15_(0u, 2u);
                        loop {
                            let _e1466_ = phi_34888_;
                            let _e1468_ = phi_2685_;
                            if (_e1468_.member < _e1468_.member_1_) {
                                phi_2686_ = type_15_((_e1468_.member + 1u), _e1468_.member_1_);
                                phi_2709_ = type_15_(1u, _e1468_.member);
                            } else {
                                phi_2686_ = _e1468_;
                                phi_2709_ = type_15_(0u, u32());
                            }
                            let _e1481_ = phi_2686_;
                            let _e1483_ = phi_2709_;
                            let _e1487_ = (bitcast<i32>(_e1483_.member) != 0i);
                            if _e1487_ {
                                let _e1489_ = ((_e1092_.member_1_ * 256u) + (_e1483_.member_1_ * 128u));
                                if (_e204_ > _e1489_) {
                                    phi_2724_ = (_e204_ - _e1489_);
                                } else {
                                    phi_2724_ = 0u;
                                }
                                let _e1493_ = phi_2724_;
                                let _e1495_ = select(_e1493_, 128u, (128u < _e1493_));
                                let _e1496_ = (_e1495_ > 0u);
                                if _e1496_ {
                                    let _e1497_ = (_e158_ < _e1495_);
                                    let _e1500_ = ((_e193_ * _e204_) + (_e1489_ + _e158_));
                                    let _e1501_ = (_e1500_ * 2u);
                                    if _e1497_ {
                                        let _e1502_ = (_e1501_ < _e150_);
                                        if _e1502_ {
                                            if _e1502_ {
                                            } else {
                                                phi_37452_ = true;
                                                break;
                                            }
                                            let _e1505_ = core_inputs.member[_e1501_];
                                            phi_2753_ = _e1505_;
                                        } else {
                                            phi_2753_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e1507_ = phi_2753_;
                                        let _e1508_ = (_e1501_ + 1u);
                                        let _e1509_ = (_e1508_ < _e150_);
                                        if _e1509_ {
                                            if _e1509_ {
                                            } else {
                                                phi_37452_ = true;
                                                break;
                                            }
                                            let _e1512_ = core_inputs.member[_e1508_];
                                            phi_2766_ = _e1512_;
                                        } else {
                                            phi_2766_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e1514_ = phi_2766_;
                                        phi_2767_ = _e1514_;
                                        phi_2768_ = _e1507_;
                                    } else {
                                        phi_2767_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        phi_2768_ = vec4<f32>(0f, 0f, 0f, 0f);
                                    }
                                    let _e1516_ = phi_2767_;
                                    let _e1518_ = phi_2768_;
                                    if _e201_ {
                                        if _e197_ {
                                            phi_35084_ = _e1466_;
                                            phi_2777_ = type_15_(0u, 4u);
                                            loop {
                                                let _e2243_ = phi_35084_;
                                                let _e2245_ = phi_2777_;
                                                if (_e2245_.member < _e2245_.member_1_) {
                                                    phi_2778_ = type_15_((_e2245_.member + 1u), _e2245_.member_1_);
                                                    phi_2801_ = type_15_(1u, _e2245_.member);
                                                } else {
                                                    phi_2778_ = _e2245_;
                                                    phi_2801_ = type_15_(0u, u32());
                                                }
                                                let _e2258_ = phi_2778_;
                                                let _e2260_ = phi_2801_;
                                                let _e2264_ = (bitcast<i32>(_e2260_.member) != 0i);
                                                if _e2264_ {
                                                    let _e2265_ = (_e2260_.member_1_ & 3u);
                                                    switch bitcast<i32>(_e2265_) {
                                                        case 0: {
                                                            phi_2914_ = _e321_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_2914_ = _e321_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_2914_ = _e321_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_2914_ = _e321_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2272_ = phi_2914_;
                                                    switch bitcast<i32>(_e2265_) {
                                                        case 0: {
                                                            phi_2931_ = _e335_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_2931_ = _e335_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_2931_ = _e335_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_2931_ = _e335_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2279_ = phi_2931_;
                                                    switch bitcast<i32>(_e2265_) {
                                                        case 0: {
                                                            phi_2942_ = _e342_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_2942_ = _e342_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_2942_ = _e342_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_2942_ = _e342_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2286_ = phi_2942_;
                                                    switch bitcast<i32>(_e2265_) {
                                                        case 0: {
                                                            phi_2953_ = _e349_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_2953_ = _e349_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_2953_ = _e349_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_2953_ = _e349_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2293_ = phi_2953_;
                                                    switch bitcast<i32>(_e2265_) {
                                                        case 0: {
                                                            phi_2964_ = _e356_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_2964_ = _e356_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_2964_ = _e356_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_2964_ = _e356_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2300_ = phi_2964_;
                                                    switch bitcast<i32>(_e2265_) {
                                                        case 0: {
                                                            phi_2975_ = _e363_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_2975_ = _e363_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_2975_ = _e363_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_2975_ = _e363_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2307_ = phi_2975_;
                                                    switch bitcast<i32>(_e2265_) {
                                                        case 0: {
                                                            phi_3190_ = _e412_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3190_ = _e412_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3190_ = _e412_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3190_ = _e412_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2314_ = phi_3190_;
                                                    switch bitcast<i32>(_e2265_) {
                                                        case 0: {
                                                            phi_3265_ = _e463_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3265_ = _e463_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3265_ = _e463_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3265_ = _e463_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2321_ = phi_3265_;
                                                    if _e1497_ {
                                                        let _e2323_ = select(_e1518_, _e1516_, vec4(_e161_));
                                                        switch bitcast<i32>(_e2265_) {
                                                            case 0: {
                                                                phi_3283_ = _e2323_.x;
                                                                break;
                                                            }
                                                            case 1: {
                                                                phi_3283_ = _e2323_.y;
                                                                break;
                                                            }
                                                            case 2: {
                                                                phi_3283_ = _e2323_.z;
                                                                break;
                                                            }
                                                            default: {
                                                                phi_3283_ = _e2323_.w;
                                                                break;
                                                            }
                                                        }
                                                        let _e2330_ = phi_3283_;
                                                        let _e2331_ = bitcast<u32>(_e2330_);
                                                        let _e2332_ = (_e2331_ & 2139095040u);
                                                        if (_e2332_ == 2139095040u) {
                                                            phi_3294_ = true;
                                                        } else {
                                                            phi_3294_ = (_e2332_ == 0u);
                                                        }
                                                        let _e2336_ = phi_3294_;
                                                        phi_3300_ = bitcast<f32>(select(_e2331_, 0u, _e2336_));
                                                    } else {
                                                        phi_3300_ = 0f;
                                                    }
                                                    let _e2340_ = phi_3300_;
                                                    if _e161_ {
                                                        phi_3309_ = (8u + _e2260_.member_1_);
                                                    } else {
                                                        phi_3309_ = _e2260_.member_1_;
                                                    }
                                                    let _e2343_ = phi_3309_;
                                                    if (_e2343_ < 256u) {
                                                    } else {
                                                        phi_35271_ = true;
                                                        break;
                                                    }
                                                    let _e2346_ = aux_scratch[_e2343_];
                                                    let _e2347_ = (_e2343_ + 4u);
                                                    if (_e2347_ < 256u) {
                                                    } else {
                                                        phi_35271_ = true;
                                                        break;
                                                    }
                                                    let _e2350_ = aux_scratch[_e2347_];
                                                    let _e2351_ = array<f32, 2>(_e2346_, _e2350_);
                                                    phi_35072_ = _e2243_;
                                                    phi_3328_ = type_15_(0u, 6u);
                                                    phi_3331_ = _e2351_;
                                                    phi_3333_ = _e2350_;
                                                    phi_3335_ = _e2346_;
                                                    phi_3337_ = _e2351_;
                                                    phi_3339_ = false;
                                                    loop {
                                                        let _e2353_ = phi_35072_;
                                                        let _e2355_ = phi_3328_;
                                                        let _e2357_ = phi_3331_;
                                                        let _e2359_ = phi_3333_;
                                                        let _e2361_ = phi_3335_;
                                                        let _e2363_ = phi_3337_;
                                                        let _e2365_ = phi_3339_;
                                                        local_18_ = _e2365_;
                                                        local_23_ = _e2361_;
                                                        local_24_ = _e2361_;
                                                        local_25_ = _e2359_;
                                                        local_26_ = _e2361_;
                                                        local_27_ = _e2361_;
                                                        local_28_ = _e2359_;
                                                        local_29_ = _e2363_;
                                                        local_30_ = _e2363_;
                                                        local_31_ = _e2363_;
                                                        local_32_ = _e2363_;
                                                        if (_e2355_.member < _e2355_.member_1_) {
                                                            phi_3329_ = type_15_((_e2355_.member + 1u), _e2355_.member_1_);
                                                            phi_3362_ = type_15_(1u, _e2355_.member);
                                                        } else {
                                                            phi_3329_ = _e2355_;
                                                            phi_3362_ = type_15_(0u, u32());
                                                        }
                                                        let _e2378_ = phi_3329_;
                                                        let _e2380_ = phi_3362_;
                                                        let _e2384_ = (bitcast<i32>(_e2380_.member) != 0i);
                                                        if _e2384_ {
                                                            if _e2365_ {
                                                                phi_35183_ = _e2353_;
                                                                phi_4459_ = _e2357_;
                                                                phi_4460_ = _e2359_;
                                                                phi_4461_ = _e2361_;
                                                                phi_4462_ = _e2363_;
                                                                phi_4463_ = _e2365_;
                                                            } else {
                                                                let _e2385_ = (_e2314_ * 0.5f);
                                                                if (_e2385_ != _e2385_) {
                                                                    phi_23871_ = true;
                                                                } else {
                                                                    phi_23871_ = (0.00001f >= _e2385_);
                                                                }
                                                                let _e2389_ = phi_23871_;
                                                                let _e2390_ = select(_e2385_, 0.00001f, _e2389_);
                                                                if (_e2272_ != _e2272_) {
                                                                    phi_23886_ = true;
                                                                } else {
                                                                    phi_23886_ = (0f >= _e2272_);
                                                                }
                                                                let _e2394_ = phi_23886_;
                                                                if (_e2321_ != _e2321_) {
                                                                    phi_23901_ = true;
                                                                } else {
                                                                    phi_23901_ = (0f >= _e2321_);
                                                                }
                                                                let _e2399_ = phi_23901_;
                                                                let _e2401_ = (_e2361_ * _e2361_);
                                                                if (_e2401_ != _e2401_) {
                                                                    phi_23916_ = true;
                                                                } else {
                                                                    phi_23916_ = (64f <= _e2401_);
                                                                }
                                                                let _e2405_ = phi_23916_;
                                                                let _e2411_ = (1f + (_e2390_ * (_e2390_ + (select(_e2272_, 0f, _e2394_) + (select(_e2321_, 0f, _e2399_) * select(_e2401_, 64f, _e2405_))))));
                                                                let _e2414_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2411_)));
                                                                let _e2418_ = (_e2414_ + (_e2414_ * (1f - (_e2411_ * _e2414_))));
                                                                let _e2422_ = (_e2418_ + (_e2418_ * (1f - (_e2411_ * _e2418_))));
                                                                let _e2423_ = (_e2390_ * _e2422_);
                                                                let _e2424_ = (_e2390_ * _e2423_);
                                                                if _e1497_ {
                                                                    let _e2425_ = (_e2423_ + _e2423_);
                                                                    let _e2426_ = (_e2424_ + _e2424_);
                                                                    phi_3410_ = type_31_(((_e2422_ + _e2422_) - 1f), -(_e2425_), _e2425_, (1f - _e2426_), (_e2425_ * _e2340_), (_e2426_ * _e2340_));
                                                                } else {
                                                                    phi_3410_ = type_31_(1f, 0f, 0f, 1f, 0f, 0f);
                                                                }
                                                                let _e2435_ = phi_3410_;
                                                                let _e2436_ = select(0u, 768u, _e161_);
                                                                let _e2437_ = (_e158_ & 127u);
                                                                let _e2439_ = (_e2436_ + _e2437_);
                                                                let _e2440_ = (_e2439_ < 1536u);
                                                                if _e2440_ {
                                                                } else {
                                                                    phi_35181_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2439_] = _e2435_.member;
                                                                let _e2443_ = (_e2436_ + 128u);
                                                                let _e2444_ = (_e2443_ + _e2437_);
                                                                let _e2445_ = (_e2444_ < 1536u);
                                                                if _e2445_ {
                                                                } else {
                                                                    phi_35181_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2444_] = _e2435_.member_1_;
                                                                let _e2448_ = (_e2436_ + 256u);
                                                                let _e2449_ = (_e2448_ + _e2437_);
                                                                let _e2450_ = (_e2449_ < 1536u);
                                                                if _e2450_ {
                                                                } else {
                                                                    phi_35181_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2449_] = _e2435_.member_2_;
                                                                let _e2453_ = (_e2436_ + 384u);
                                                                let _e2454_ = (_e2453_ + _e2437_);
                                                                let _e2455_ = (_e2454_ < 1536u);
                                                                if _e2455_ {
                                                                } else {
                                                                    phi_35181_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2454_] = _e2435_.member_3_;
                                                                let _e2458_ = (_e2436_ + 512u);
                                                                let _e2459_ = (_e2458_ + _e2437_);
                                                                let _e2460_ = (_e2459_ < 1536u);
                                                                if _e2460_ {
                                                                } else {
                                                                    phi_35181_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2459_] = _e2435_.member_4_;
                                                                let _e2463_ = (_e2436_ + 640u);
                                                                let _e2464_ = (_e2463_ + _e2437_);
                                                                let _e2465_ = (_e2464_ < 1536u);
                                                                if _e2465_ {
                                                                } else {
                                                                    phi_35181_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2464_] = _e2435_.member_5_;
                                                                workgroupBarrier();
                                                                phi_3473_ = 1u;
                                                                phi_3476_ = _e2435_;
                                                                loop {
                                                                    let _e2468_ = phi_3473_;
                                                                    let _e2470_ = phi_3476_;
                                                                    let _e2471_ = (_e2468_ < 128u);
                                                                    if _e2471_ {
                                                                        if (_e158_ >= _e2468_) {
                                                                            let _e2474_ = ((_e158_ - _e2468_) & 127u);
                                                                            let _e2475_ = (_e2436_ + _e2474_);
                                                                            if (_e2475_ < 1536u) {
                                                                            } else {
                                                                                phi_35058_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2478_ = scan_scratch[_e2475_];
                                                                            let _e2479_ = (_e2443_ + _e2474_);
                                                                            if (_e2479_ < 1536u) {
                                                                            } else {
                                                                                phi_35058_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2482_ = scan_scratch[_e2479_];
                                                                            let _e2483_ = (_e2448_ + _e2474_);
                                                                            if (_e2483_ < 1536u) {
                                                                            } else {
                                                                                phi_35058_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2486_ = scan_scratch[_e2483_];
                                                                            let _e2487_ = (_e2453_ + _e2474_);
                                                                            if (_e2487_ < 1536u) {
                                                                            } else {
                                                                                phi_35058_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2490_ = scan_scratch[_e2487_];
                                                                            let _e2491_ = (_e2458_ + _e2474_);
                                                                            if (_e2491_ < 1536u) {
                                                                            } else {
                                                                                phi_35058_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2494_ = scan_scratch[_e2491_];
                                                                            let _e2495_ = (_e2463_ + _e2474_);
                                                                            if (_e2495_ < 1536u) {
                                                                            } else {
                                                                                phi_35058_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2498_ = scan_scratch[_e2495_];
                                                                            phi_3578_ = type_31_(((_e2470_.member * _e2478_) + (_e2470_.member_1_ * _e2486_)), ((_e2470_.member * _e2482_) + (_e2470_.member_1_ * _e2490_)), ((_e2470_.member_2_ * _e2478_) + (_e2470_.member_3_ * _e2486_)), ((_e2470_.member_2_ * _e2482_) + (_e2470_.member_3_ * _e2490_)), (((_e2470_.member * _e2494_) + (_e2470_.member_1_ * _e2498_)) + _e2470_.member_4_), (((_e2470_.member_2_ * _e2494_) + (_e2470_.member_3_ * _e2498_)) + _e2470_.member_5_));
                                                                        } else {
                                                                            phi_3578_ = _e2470_;
                                                                        }
                                                                        let _e2527_ = phi_3578_;
                                                                        workgroupBarrier();
                                                                        if _e2440_ {
                                                                        } else {
                                                                            phi_35058_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2439_] = _e2527_.member;
                                                                        if _e2445_ {
                                                                        } else {
                                                                            phi_35058_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2444_] = _e2527_.member_1_;
                                                                        if _e2450_ {
                                                                        } else {
                                                                            phi_35058_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2449_] = _e2527_.member_2_;
                                                                        if _e2455_ {
                                                                        } else {
                                                                            phi_35058_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2454_] = _e2527_.member_3_;
                                                                        if _e2460_ {
                                                                        } else {
                                                                            phi_35058_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2459_] = _e2527_.member_4_;
                                                                        if _e2465_ {
                                                                        } else {
                                                                            phi_35058_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2464_] = _e2527_.member_5_;
                                                                        workgroupBarrier();
                                                                        phi_3474_ = (_e2468_ << bitcast<u32>(1i));
                                                                        phi_3477_ = _e2527_;
                                                                    } else {
                                                                        phi_3474_ = u32();
                                                                        phi_3477_ = type_31_();
                                                                    }
                                                                    let _e2537_ = phi_3474_;
                                                                    let _e2539_ = phi_3477_;
                                                                    phi_3473_ = _e2537_;
                                                                    phi_3476_ = _e2539_;
                                                                    phi_35058_ = _e2353_;
                                                                    if !(_e2471_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e2542_ = phi_35058_;
                                                                phi_35181_ = _e2542_;
                                                                if _e2542_ {
                                                                    break;
                                                                }
                                                                if (_e158_ == 0u) {
                                                                    phi_3702_ = _e2351_;
                                                                } else {
                                                                    let _e2545_ = ((_e158_ - 1u) & 127u);
                                                                    let _e2546_ = (_e2436_ + _e2545_);
                                                                    if (_e2546_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2549_ = scan_scratch[_e2546_];
                                                                    let _e2550_ = (_e2443_ + _e2545_);
                                                                    if (_e2550_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2553_ = scan_scratch[_e2550_];
                                                                    let _e2554_ = (_e2448_ + _e2545_);
                                                                    if (_e2554_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2557_ = scan_scratch[_e2554_];
                                                                    let _e2558_ = (_e2453_ + _e2545_);
                                                                    if (_e2558_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2561_ = scan_scratch[_e2558_];
                                                                    let _e2562_ = (_e2458_ + _e2545_);
                                                                    if (_e2562_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2565_ = scan_scratch[_e2562_];
                                                                    let _e2566_ = (_e2463_ + _e2545_);
                                                                    if (_e2566_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2569_ = scan_scratch[_e2566_];
                                                                    phi_3702_ = array<f32, 2>((((_e2549_ * _e2346_) + (_e2553_ * _e2350_)) + _e2565_), (((_e2557_ * _e2346_) + (_e2561_ * _e2350_)) + _e2569_));
                                                                }
                                                                let _e2580_ = phi_3702_;
                                                                if _e1496_ {
                                                                    if (_e1495_ <= 128u) {
                                                                        phi_3714_ = (_e1495_ - 1u);
                                                                    } else {
                                                                        phi_3714_ = 127u;
                                                                    }
                                                                    let _e2586_ = phi_3714_;
                                                                    let _e2587_ = (_e2586_ & 127u);
                                                                    let _e2588_ = (_e2436_ + _e2587_);
                                                                    if (_e2588_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2591_ = scan_scratch[_e2588_];
                                                                    let _e2592_ = (_e2443_ + _e2587_);
                                                                    if (_e2592_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2595_ = scan_scratch[_e2592_];
                                                                    let _e2596_ = (_e2448_ + _e2587_);
                                                                    if (_e2596_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2599_ = scan_scratch[_e2596_];
                                                                    let _e2600_ = (_e2453_ + _e2587_);
                                                                    if (_e2600_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2603_ = scan_scratch[_e2600_];
                                                                    let _e2604_ = (_e2458_ + _e2587_);
                                                                    if (_e2604_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2607_ = scan_scratch[_e2604_];
                                                                    let _e2608_ = (_e2463_ + _e2587_);
                                                                    if (_e2608_ < 1536u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2611_ = scan_scratch[_e2608_];
                                                                    phi_3778_ = array<f32, 2>((((_e2591_ * _e2346_) + (_e2595_ * _e2350_)) + _e2607_), (((_e2599_ * _e2346_) + (_e2603_ * _e2350_)) + _e2611_));
                                                                } else {
                                                                    phi_3778_ = _e2351_;
                                                                }
                                                                let _e2622_ = phi_3778_;
                                                                workgroupBarrier();
                                                                if _e1497_ {
                                                                    if ((bitcast<u32>(_e2580_[0]) & 2139095040u) != 2139095040u) {
                                                                        let _e2628_ = ((bitcast<u32>(_e2361_) & 2139095040u) != 2139095040u);
                                                                        if _e2628_ {
                                                                            let _e2629_ = abs(_e2580_[0]);
                                                                            let _e2630_ = abs(_e2361_);
                                                                            let _e2632_ = select(_e2630_, _e2629_, (_e2629_ > _e2630_));
                                                                            let _e2634_ = select(1f, _e2632_, (_e2632_ > 1f));
                                                                            if ((bitcast<u32>(_e2634_) & 2139095040u) != 2139095040u) {
                                                                                let _e2640_ = (_e2634_ <= 0f);
                                                                                if _e2640_ {
                                                                                    phi_3900_ = f32();
                                                                                } else {
                                                                                    if (_e2634_ < 0.00001f) {
                                                                                        let _e2672_ = (select(0.000000000000000000000001f, _e2634_, (_e2634_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                        if (_e2672_ != _e2672_) {
                                                                                            phi_23991_ = true;
                                                                                        } else {
                                                                                            phi_23991_ = (0.000001f >= _e2672_);
                                                                                        }
                                                                                        let _e2676_ = phi_23991_;
                                                                                        let _e2677_ = select(_e2672_, 0.000001f, _e2676_);
                                                                                        let _e2680_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2677_)));
                                                                                        let _e2684_ = (_e2680_ + (_e2680_ * (1f - (_e2677_ * _e2680_))));
                                                                                        let _e2688_ = (_e2684_ + (_e2684_ * (1f - (_e2677_ * _e2684_))));
                                                                                        let _e2694_ = bitcast<u32>(((_e2688_ + (_e2688_ * (1f - (_e2677_ * _e2688_)))) * 18446744000000000000f));
                                                                                        let _e2695_ = (_e2694_ & 2139095040u);
                                                                                        if (_e2695_ == 2139095040u) {
                                                                                            phi_3858_ = true;
                                                                                        } else {
                                                                                            phi_3858_ = (_e2695_ == 0u);
                                                                                        }
                                                                                        let _e2699_ = phi_3858_;
                                                                                        phi_3899_ = bitcast<f32>(select(_e2694_, 0u, _e2699_));
                                                                                    } else {
                                                                                        if (_e2634_ != _e2634_) {
                                                                                            phi_23931_ = true;
                                                                                        } else {
                                                                                            phi_23931_ = (0.000001f >= _e2634_);
                                                                                        }
                                                                                        let _e2645_ = phi_23931_;
                                                                                        let _e2646_ = select(_e2634_, 0.000001f, _e2645_);
                                                                                        let _e2649_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2646_)));
                                                                                        let _e2653_ = (_e2649_ + (_e2649_ * (1f - (_e2646_ * _e2649_))));
                                                                                        let _e2657_ = (_e2653_ + (_e2653_ * (1f - (_e2646_ * _e2653_))));
                                                                                        let _e2662_ = bitcast<u32>((_e2657_ + (_e2657_ * (1f - (_e2646_ * _e2657_)))));
                                                                                        let _e2663_ = (_e2662_ & 2139095040u);
                                                                                        if (_e2663_ == 2139095040u) {
                                                                                            phi_3893_ = true;
                                                                                        } else {
                                                                                            phi_3893_ = (_e2663_ == 0u);
                                                                                        }
                                                                                        let _e2667_ = phi_3893_;
                                                                                        phi_3899_ = bitcast<f32>(select(_e2662_, 0u, _e2667_));
                                                                                    }
                                                                                    let _e2703_ = phi_3899_;
                                                                                    phi_3900_ = _e2703_;
                                                                                }
                                                                                let _e2705_ = phi_3900_;
                                                                                phi_3901_ = _e2705_;
                                                                                phi_3902_ = _e2640_;
                                                                            } else {
                                                                                phi_3901_ = f32();
                                                                                phi_3902_ = true;
                                                                            }
                                                                            let _e2707_ = phi_3901_;
                                                                            let _e2709_ = phi_3902_;
                                                                            phi_3908_ = (abs((_e2580_[0] - _e2361_)) * select(_e2707_, 0f, _e2709_));
                                                                        } else {
                                                                            phi_3908_ = f32();
                                                                        }
                                                                        let _e2713_ = phi_3908_;
                                                                        phi_3910_ = _e2713_;
                                                                        phi_3911_ = select(true, false, _e2628_);
                                                                    } else {
                                                                        phi_3910_ = f32();
                                                                        phi_3911_ = true;
                                                                    }
                                                                    let _e2716_ = phi_3910_;
                                                                    let _e2718_ = phi_3911_;
                                                                    let _e2719_ = select(_e2716_, 1000000000f, _e2718_);
                                                                    if ((bitcast<u32>(_e2580_[1]) & 2139095040u) != 2139095040u) {
                                                                        let _e2725_ = ((bitcast<u32>(_e2359_) & 2139095040u) != 2139095040u);
                                                                        if _e2725_ {
                                                                            let _e2726_ = abs(_e2580_[1]);
                                                                            let _e2727_ = abs(_e2359_);
                                                                            let _e2729_ = select(_e2727_, _e2726_, (_e2726_ > _e2727_));
                                                                            let _e2731_ = select(1f, _e2729_, (_e2729_ > 1f));
                                                                            if ((bitcast<u32>(_e2731_) & 2139095040u) != 2139095040u) {
                                                                                let _e2737_ = (_e2731_ <= 0f);
                                                                                if _e2737_ {
                                                                                    phi_4034_ = f32();
                                                                                } else {
                                                                                    if (_e2731_ < 0.00001f) {
                                                                                        let _e2769_ = (select(0.000000000000000000000001f, _e2731_, (_e2731_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                        if (_e2769_ != _e2769_) {
                                                                                            phi_24111_ = true;
                                                                                        } else {
                                                                                            phi_24111_ = (0.000001f >= _e2769_);
                                                                                        }
                                                                                        let _e2773_ = phi_24111_;
                                                                                        let _e2774_ = select(_e2769_, 0.000001f, _e2773_);
                                                                                        let _e2777_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2774_)));
                                                                                        let _e2781_ = (_e2777_ + (_e2777_ * (1f - (_e2774_ * _e2777_))));
                                                                                        let _e2785_ = (_e2781_ + (_e2781_ * (1f - (_e2774_ * _e2781_))));
                                                                                        let _e2791_ = bitcast<u32>(((_e2785_ + (_e2785_ * (1f - (_e2774_ * _e2785_)))) * 18446744000000000000f));
                                                                                        let _e2792_ = (_e2791_ & 2139095040u);
                                                                                        if (_e2792_ == 2139095040u) {
                                                                                            phi_3992_ = true;
                                                                                        } else {
                                                                                            phi_3992_ = (_e2792_ == 0u);
                                                                                        }
                                                                                        let _e2796_ = phi_3992_;
                                                                                        phi_4033_ = bitcast<f32>(select(_e2791_, 0u, _e2796_));
                                                                                    } else {
                                                                                        if (_e2731_ != _e2731_) {
                                                                                            phi_24051_ = true;
                                                                                        } else {
                                                                                            phi_24051_ = (0.000001f >= _e2731_);
                                                                                        }
                                                                                        let _e2742_ = phi_24051_;
                                                                                        let _e2743_ = select(_e2731_, 0.000001f, _e2742_);
                                                                                        let _e2746_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2743_)));
                                                                                        let _e2750_ = (_e2746_ + (_e2746_ * (1f - (_e2743_ * _e2746_))));
                                                                                        let _e2754_ = (_e2750_ + (_e2750_ * (1f - (_e2743_ * _e2750_))));
                                                                                        let _e2759_ = bitcast<u32>((_e2754_ + (_e2754_ * (1f - (_e2743_ * _e2754_)))));
                                                                                        let _e2760_ = (_e2759_ & 2139095040u);
                                                                                        if (_e2760_ == 2139095040u) {
                                                                                            phi_4027_ = true;
                                                                                        } else {
                                                                                            phi_4027_ = (_e2760_ == 0u);
                                                                                        }
                                                                                        let _e2764_ = phi_4027_;
                                                                                        phi_4033_ = bitcast<f32>(select(_e2759_, 0u, _e2764_));
                                                                                    }
                                                                                    let _e2800_ = phi_4033_;
                                                                                    phi_4034_ = _e2800_;
                                                                                }
                                                                                let _e2802_ = phi_4034_;
                                                                                phi_4035_ = _e2802_;
                                                                                phi_4036_ = _e2737_;
                                                                            } else {
                                                                                phi_4035_ = f32();
                                                                                phi_4036_ = true;
                                                                            }
                                                                            let _e2804_ = phi_4035_;
                                                                            let _e2806_ = phi_4036_;
                                                                            phi_4042_ = (abs((_e2580_[1] - _e2359_)) * select(_e2804_, 0f, _e2806_));
                                                                        } else {
                                                                            phi_4042_ = f32();
                                                                        }
                                                                        let _e2810_ = phi_4042_;
                                                                        phi_4044_ = _e2810_;
                                                                        phi_4045_ = select(true, false, _e2725_);
                                                                    } else {
                                                                        phi_4044_ = f32();
                                                                        phi_4045_ = true;
                                                                    }
                                                                    let _e2813_ = phi_4044_;
                                                                    let _e2815_ = phi_4045_;
                                                                    let _e2816_ = select(_e2813_, 1000000000f, _e2815_);
                                                                    let _e2818_ = select(_e2719_, _e2816_, (_e2816_ > _e2719_));
                                                                    if (_e2380_.member_1_ > 0u) {
                                                                        let _e2821_ = ((_e158_ + 1u) == _e1495_);
                                                                        if _e2821_ {
                                                                            if ((bitcast<u32>(_e2622_[0]) & 2139095040u) != 2139095040u) {
                                                                                let _e2829_ = ((bitcast<u32>(_e2357_[0]) & 2139095040u) != 2139095040u);
                                                                                if _e2829_ {
                                                                                    let _e2830_ = abs(_e2622_[0]);
                                                                                    let _e2831_ = abs(_e2357_[0]);
                                                                                    let _e2833_ = select(_e2831_, _e2830_, (_e2830_ > _e2831_));
                                                                                    let _e2835_ = select(1f, _e2833_, (_e2833_ > 1f));
                                                                                    if ((bitcast<u32>(_e2835_) & 2139095040u) != 2139095040u) {
                                                                                        let _e2841_ = (_e2835_ <= 0f);
                                                                                        if _e2841_ {
                                                                                            phi_4184_ = f32();
                                                                                        } else {
                                                                                            if (_e2835_ < 0.00001f) {
                                                                                                let _e2873_ = (select(0.000000000000000000000001f, _e2835_, (_e2835_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                                if (_e2873_ != _e2873_) {
                                                                                                    phi_24231_ = true;
                                                                                                } else {
                                                                                                    phi_24231_ = (0.000001f >= _e2873_);
                                                                                                }
                                                                                                let _e2877_ = phi_24231_;
                                                                                                let _e2878_ = select(_e2873_, 0.000001f, _e2877_);
                                                                                                let _e2881_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2878_)));
                                                                                                let _e2885_ = (_e2881_ + (_e2881_ * (1f - (_e2878_ * _e2881_))));
                                                                                                let _e2889_ = (_e2885_ + (_e2885_ * (1f - (_e2878_ * _e2885_))));
                                                                                                let _e2895_ = bitcast<u32>(((_e2889_ + (_e2889_ * (1f - (_e2878_ * _e2889_)))) * 18446744000000000000f));
                                                                                                let _e2896_ = (_e2895_ & 2139095040u);
                                                                                                if (_e2896_ == 2139095040u) {
                                                                                                    phi_4142_ = true;
                                                                                                } else {
                                                                                                    phi_4142_ = (_e2896_ == 0u);
                                                                                                }
                                                                                                let _e2900_ = phi_4142_;
                                                                                                phi_4183_ = bitcast<f32>(select(_e2895_, 0u, _e2900_));
                                                                                            } else {
                                                                                                if (_e2835_ != _e2835_) {
                                                                                                    phi_24171_ = true;
                                                                                                } else {
                                                                                                    phi_24171_ = (0.000001f >= _e2835_);
                                                                                                }
                                                                                                let _e2846_ = phi_24171_;
                                                                                                let _e2847_ = select(_e2835_, 0.000001f, _e2846_);
                                                                                                let _e2850_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2847_)));
                                                                                                let _e2854_ = (_e2850_ + (_e2850_ * (1f - (_e2847_ * _e2850_))));
                                                                                                let _e2858_ = (_e2854_ + (_e2854_ * (1f - (_e2847_ * _e2854_))));
                                                                                                let _e2863_ = bitcast<u32>((_e2858_ + (_e2858_ * (1f - (_e2847_ * _e2858_)))));
                                                                                                let _e2864_ = (_e2863_ & 2139095040u);
                                                                                                if (_e2864_ == 2139095040u) {
                                                                                                    phi_4177_ = true;
                                                                                                } else {
                                                                                                    phi_4177_ = (_e2864_ == 0u);
                                                                                                }
                                                                                                let _e2868_ = phi_4177_;
                                                                                                phi_4183_ = bitcast<f32>(select(_e2863_, 0u, _e2868_));
                                                                                            }
                                                                                            let _e2904_ = phi_4183_;
                                                                                            phi_4184_ = _e2904_;
                                                                                        }
                                                                                        let _e2906_ = phi_4184_;
                                                                                        phi_4185_ = _e2906_;
                                                                                        phi_4186_ = _e2841_;
                                                                                    } else {
                                                                                        phi_4185_ = f32();
                                                                                        phi_4186_ = true;
                                                                                    }
                                                                                    let _e2908_ = phi_4185_;
                                                                                    let _e2910_ = phi_4186_;
                                                                                    phi_4192_ = (abs((_e2622_[0] - _e2357_[0])) * select(_e2908_, 0f, _e2910_));
                                                                                } else {
                                                                                    phi_4192_ = f32();
                                                                                }
                                                                                let _e2914_ = phi_4192_;
                                                                                phi_4194_ = _e2914_;
                                                                                phi_4195_ = select(true, false, _e2829_);
                                                                            } else {
                                                                                phi_4194_ = f32();
                                                                                phi_4195_ = true;
                                                                            }
                                                                            let _e2917_ = phi_4194_;
                                                                            let _e2919_ = phi_4195_;
                                                                            let _e2920_ = select(_e2917_, 1000000000f, _e2919_);
                                                                            if ((bitcast<u32>(_e2622_[1]) & 2139095040u) != 2139095040u) {
                                                                                let _e2928_ = ((bitcast<u32>(_e2357_[1]) & 2139095040u) != 2139095040u);
                                                                                if _e2928_ {
                                                                                    let _e2929_ = abs(_e2622_[1]);
                                                                                    let _e2930_ = abs(_e2357_[1]);
                                                                                    let _e2932_ = select(_e2930_, _e2929_, (_e2929_ > _e2930_));
                                                                                    let _e2934_ = select(1f, _e2932_, (_e2932_ > 1f));
                                                                                    if ((bitcast<u32>(_e2934_) & 2139095040u) != 2139095040u) {
                                                                                        let _e2940_ = (_e2934_ <= 0f);
                                                                                        if _e2940_ {
                                                                                            phi_4320_ = f32();
                                                                                        } else {
                                                                                            if (_e2934_ < 0.00001f) {
                                                                                                let _e2972_ = (select(0.000000000000000000000001f, _e2934_, (_e2934_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                                if (_e2972_ != _e2972_) {
                                                                                                    phi_24351_ = true;
                                                                                                } else {
                                                                                                    phi_24351_ = (0.000001f >= _e2972_);
                                                                                                }
                                                                                                let _e2976_ = phi_24351_;
                                                                                                let _e2977_ = select(_e2972_, 0.000001f, _e2976_);
                                                                                                let _e2980_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2977_)));
                                                                                                let _e2984_ = (_e2980_ + (_e2980_ * (1f - (_e2977_ * _e2980_))));
                                                                                                let _e2988_ = (_e2984_ + (_e2984_ * (1f - (_e2977_ * _e2984_))));
                                                                                                let _e2994_ = bitcast<u32>(((_e2988_ + (_e2988_ * (1f - (_e2977_ * _e2988_)))) * 18446744000000000000f));
                                                                                                let _e2995_ = (_e2994_ & 2139095040u);
                                                                                                if (_e2995_ == 2139095040u) {
                                                                                                    phi_4278_ = true;
                                                                                                } else {
                                                                                                    phi_4278_ = (_e2995_ == 0u);
                                                                                                }
                                                                                                let _e2999_ = phi_4278_;
                                                                                                phi_4319_ = bitcast<f32>(select(_e2994_, 0u, _e2999_));
                                                                                            } else {
                                                                                                if (_e2934_ != _e2934_) {
                                                                                                    phi_24291_ = true;
                                                                                                } else {
                                                                                                    phi_24291_ = (0.000001f >= _e2934_);
                                                                                                }
                                                                                                let _e2945_ = phi_24291_;
                                                                                                let _e2946_ = select(_e2934_, 0.000001f, _e2945_);
                                                                                                let _e2949_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2946_)));
                                                                                                let _e2953_ = (_e2949_ + (_e2949_ * (1f - (_e2946_ * _e2949_))));
                                                                                                let _e2957_ = (_e2953_ + (_e2953_ * (1f - (_e2946_ * _e2953_))));
                                                                                                let _e2962_ = bitcast<u32>((_e2957_ + (_e2957_ * (1f - (_e2946_ * _e2957_)))));
                                                                                                let _e2963_ = (_e2962_ & 2139095040u);
                                                                                                if (_e2963_ == 2139095040u) {
                                                                                                    phi_4313_ = true;
                                                                                                } else {
                                                                                                    phi_4313_ = (_e2963_ == 0u);
                                                                                                }
                                                                                                let _e2967_ = phi_4313_;
                                                                                                phi_4319_ = bitcast<f32>(select(_e2962_, 0u, _e2967_));
                                                                                            }
                                                                                            let _e3003_ = phi_4319_;
                                                                                            phi_4320_ = _e3003_;
                                                                                        }
                                                                                        let _e3005_ = phi_4320_;
                                                                                        phi_4321_ = _e3005_;
                                                                                        phi_4322_ = _e2940_;
                                                                                    } else {
                                                                                        phi_4321_ = f32();
                                                                                        phi_4322_ = true;
                                                                                    }
                                                                                    let _e3007_ = phi_4321_;
                                                                                    let _e3009_ = phi_4322_;
                                                                                    phi_4328_ = (abs((_e2622_[1] - _e2357_[1])) * select(_e3007_, 0f, _e3009_));
                                                                                } else {
                                                                                    phi_4328_ = f32();
                                                                                }
                                                                                let _e3013_ = phi_4328_;
                                                                                phi_4330_ = _e3013_;
                                                                                phi_4331_ = select(true, false, _e2928_);
                                                                            } else {
                                                                                phi_4330_ = f32();
                                                                                phi_4331_ = true;
                                                                            }
                                                                            let _e3016_ = phi_4330_;
                                                                            let _e3018_ = phi_4331_;
                                                                            let _e3019_ = select(_e3016_, 1000000000f, _e3018_);
                                                                            let _e3021_ = select(_e2920_, _e3019_, (_e3019_ > _e2920_));
                                                                            phi_4346_ = select(_e2818_, _e3021_, (_e3021_ > _e2818_));
                                                                        } else {
                                                                            phi_4346_ = f32();
                                                                        }
                                                                        let _e3025_ = phi_4346_;
                                                                        phi_4348_ = _e3025_;
                                                                        phi_4349_ = select(true, false, _e2821_);
                                                                    } else {
                                                                        phi_4348_ = f32();
                                                                        phi_4349_ = true;
                                                                    }
                                                                    let _e3028_ = phi_4348_;
                                                                    let _e3030_ = phi_4349_;
                                                                    phi_4354_ = select(_e3028_, _e2818_, _e3030_);
                                                                } else {
                                                                    phi_4354_ = 0f;
                                                                }
                                                                let _e3033_ = phi_4354_;
                                                                let _e3034_ = (_e157_ < 128u);
                                                                if _e3034_ {
                                                                    let _e3035_ = (32u + _e158_);
                                                                    if (_e3035_ < 256u) {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e3035_] = _e3033_;
                                                                }
                                                                workgroupBarrier();
                                                                if (_e157_ >= 128u) {
                                                                    let _e3039_ = (32u + _e158_);
                                                                    let _e3040_ = (_e3039_ < 256u);
                                                                    if _e3040_ {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    let _e3042_ = aux_scratch[_e3039_];
                                                                    if _e3040_ {
                                                                    } else {
                                                                        phi_35181_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e3039_] = select(_e3042_, _e3033_, (_e3033_ > _e3042_));
                                                                }
                                                                workgroupBarrier();
                                                                phi_4392_ = 64u;
                                                                loop {
                                                                    let _e3046_ = phi_4392_;
                                                                    let _e3047_ = (_e3046_ > 0u);
                                                                    if _e3047_ {
                                                                        if (_e158_ < _e3046_) {
                                                                            if _e3034_ {
                                                                                let _e3049_ = (32u + _e158_);
                                                                                let _e3050_ = (_e3049_ < 256u);
                                                                                if _e3050_ {
                                                                                } else {
                                                                                    phi_35090_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e3052_ = aux_scratch[_e3049_];
                                                                                let _e3053_ = (_e3049_ + _e3046_);
                                                                                if (_e3053_ < 256u) {
                                                                                } else {
                                                                                    phi_35090_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e3056_ = aux_scratch[_e3053_];
                                                                                if _e3050_ {
                                                                                } else {
                                                                                    phi_35090_ = true;
                                                                                    break;
                                                                                }
                                                                                aux_scratch[_e3049_] = select(_e3052_, _e3056_, (_e3056_ > _e3052_));
                                                                            }
                                                                        }
                                                                        workgroupBarrier();
                                                                        phi_4393_ = (_e3046_ >> bitcast<u32>(1i));
                                                                    } else {
                                                                        phi_4393_ = u32();
                                                                    }
                                                                    let _e3062_ = phi_4393_;
                                                                    phi_4392_ = _e3062_;
                                                                    phi_35090_ = _e2542_;
                                                                    if !(_e3047_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e3065_ = phi_35090_;
                                                                phi_35181_ = _e3065_;
                                                                if _e3065_ {
                                                                    break;
                                                                }
                                                                let _e3067_ = aux_scratch[32];
                                                                workgroupBarrier();
                                                                if (_e3067_ <= 0.0000005f) {
                                                                    if (_e2380_.member_1_ > 0u) {
                                                                        phi_4453_ = true;
                                                                    } else {
                                                                        phi_4453_ = (_e3067_ == 0f);
                                                                    }
                                                                    let _e3072_ = phi_4453_;
                                                                    phi_4458_ = select(_e2365_, true, _e3072_);
                                                                } else {
                                                                    phi_4458_ = _e2365_;
                                                                }
                                                                let _e3075_ = phi_4458_;
                                                                phi_35183_ = _e3065_;
                                                                phi_4459_ = _e2622_;
                                                                phi_4460_ = _e2580_[1];
                                                                phi_4461_ = _e2580_[0];
                                                                phi_4462_ = _e2622_;
                                                                phi_4463_ = _e3075_;
                                                            }
                                                            let _e3077_ = phi_35183_;
                                                            let _e3079_ = phi_4459_;
                                                            let _e3081_ = phi_4460_;
                                                            let _e3083_ = phi_4461_;
                                                            let _e3085_ = phi_4462_;
                                                            let _e3087_ = phi_4463_;
                                                            phi_35182_ = _e3077_;
                                                            phi_3332_ = _e3079_;
                                                            phi_3334_ = _e3081_;
                                                            phi_3336_ = _e3083_;
                                                            phi_3338_ = _e3085_;
                                                            phi_3340_ = _e3087_;
                                                        } else {
                                                            phi_35182_ = _e2353_;
                                                            phi_3332_ = array<f32, 2>();
                                                            phi_3334_ = f32();
                                                            phi_3336_ = f32();
                                                            phi_3338_ = array<f32, 2>();
                                                            phi_3340_ = bool();
                                                        }
                                                        let _e3089_ = phi_35182_;
                                                        let _e3091_ = phi_3332_;
                                                        let _e3093_ = phi_3334_;
                                                        let _e3095_ = phi_3336_;
                                                        let _e3097_ = phi_3338_;
                                                        let _e3099_ = phi_3340_;
                                                        phi_35072_ = _e3089_;
                                                        phi_3328_ = _e2378_;
                                                        phi_3331_ = _e3091_;
                                                        phi_3333_ = _e3093_;
                                                        phi_3335_ = _e3095_;
                                                        phi_3337_ = _e3097_;
                                                        phi_3339_ = _e3099_;
                                                        phi_35181_ = _e3089_;
                                                        if !(_e2384_) {
                                                            break;
                                                        }
                                                    }
                                                    let _e3102_ = phi_35181_;
                                                    phi_35271_ = _e3102_;
                                                    if _e3102_ {
                                                        break;
                                                    }
                                                    let _e3104_ = local_18_;
                                                    if _e3104_ {
                                                        if _e1497_ {
                                                            let _e3413_ = (_e2314_ * 0.5f);
                                                            if (_e3413_ != _e3413_) {
                                                                phi_24891_ = true;
                                                            } else {
                                                                phi_24891_ = (0.00001f >= _e3413_);
                                                            }
                                                            let _e3417_ = phi_24891_;
                                                            let _e3418_ = select(_e3413_, 0.00001f, _e3417_);
                                                            if (_e2272_ != _e2272_) {
                                                                phi_24906_ = true;
                                                            } else {
                                                                phi_24906_ = (0f >= _e2272_);
                                                            }
                                                            let _e3422_ = phi_24906_;
                                                            if (_e2321_ != _e2321_) {
                                                                phi_24921_ = true;
                                                            } else {
                                                                phi_24921_ = (0f >= _e2321_);
                                                            }
                                                            let _e3427_ = phi_24921_;
                                                            let _e3430_ = local_23_;
                                                            let _e3432_ = local_24_;
                                                            let _e3433_ = (_e3430_ * _e3432_);
                                                            if (_e3433_ != _e3433_) {
                                                                phi_24936_ = true;
                                                            } else {
                                                                phi_24936_ = (64f <= _e3433_);
                                                            }
                                                            let _e3437_ = phi_24936_;
                                                            let _e3440_ = (select(_e2272_, 0f, _e3422_) + (select(_e2321_, 0f, _e3427_) * select(_e3433_, 64f, _e3437_)));
                                                            let _e3443_ = (1f + (_e3418_ * (_e3418_ + _e3440_)));
                                                            let _e3446_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3443_)));
                                                            let _e3450_ = (_e3446_ + (_e3446_ * (1f - (_e3443_ * _e3446_))));
                                                            let _e3454_ = (_e3450_ + (_e3450_ * (1f - (_e3443_ * _e3450_))));
                                                            let _e3455_ = (_e3418_ * _e3454_);
                                                            let _e3459_ = local_25_;
                                                            let _e3460_ = (_e2340_ - _e3459_);
                                                            let _e3462_ = local_26_;
                                                            let _e3465_ = ((_e3454_ * _e3462_) + (_e3455_ * _e3460_));
                                                            let _e3467_ = local_27_;
                                                            let _e3470_ = local_28_;
                                                            let _e3473_ = ((_e3470_ + (_e3455_ * _e3467_)) + ((_e3418_ * _e3455_) * _e3460_));
                                                            let _e3488_ = select(0u, 2u, _e161_);
                                                            let _e3489_ = (_e2260_.member_1_ * 4u);
                                                            let _e3492_ = (_e158_ & 127u);
                                                            let _e3493_ = (((_e3489_ + _e3488_) * 128u) + _e3492_);
                                                            if (_e3493_ < 2048u) {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            lane_buf[_e3493_] = (((((_e3473_ * _e2279_) + (_e3465_ * _e2286_)) + (((_e2340_ - (_e3440_ * _e3465_)) - _e3473_) * _e2293_)) + ((_e2340_ - ((2f * _e3440_) * _e3465_)) * _e2300_)) + (_e2340_ * _e2307_));
                                                            let _e3499_ = (((_e3489_ + (_e3488_ + 1u)) * 128u) + _e3492_);
                                                            if (_e3499_ < 2048u) {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            lane_buf[_e3499_] = _e3465_;
                                                        }
                                                        if _e1098_ {
                                                            let _e3503_ = local_29_;
                                                            if (_e2260_.member_1_ < 256u) {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e2260_.member_1_] = _e3503_[0];
                                                            let _e3508_ = local_30_;
                                                            let _e3510_ = (4u + _e2260_.member_1_);
                                                            if (_e3510_ < 256u) {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e3510_] = _e3508_[1];
                                                        }
                                                        if (_e157_ == 128u) {
                                                            let _e3515_ = local_31_;
                                                            let _e3517_ = (8u + _e2260_.member_1_);
                                                            if (_e3517_ < 256u) {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e3517_] = _e3515_[0];
                                                            let _e3521_ = local_32_;
                                                            let _e3523_ = (12u + _e2260_.member_1_);
                                                            if (_e3523_ < 256u) {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e3523_] = _e3521_[1];
                                                        }
                                                        workgroupBarrier();
                                                        phi_35273_ = _e3102_;
                                                    } else {
                                                        let _e3105_ = select(0u, 128u, _e161_);
                                                        if _e1497_ {
                                                            let _e3110_ = (_e3105_ + _e158_);
                                                            if (_e3110_ < 1536u) {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            scan_scratch[_e3110_] = _e2340_;
                                                        } else {
                                                            if (_e158_ < 128u) {
                                                                let _e3107_ = (_e3105_ + _e158_);
                                                                if (_e3107_ < 1536u) {
                                                                } else {
                                                                    phi_35271_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e3107_] = 0f;
                                                            }
                                                        }
                                                        workgroupBarrier();
                                                        if _e1098_ {
                                                            let _e3113_ = (_e2260_.member_1_ < 256u);
                                                            if _e3113_ {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            let _e3115_ = aux_scratch[_e2260_.member_1_];
                                                            let _e3116_ = (4u + _e2260_.member_1_);
                                                            let _e3117_ = (_e3116_ < 256u);
                                                            if _e3117_ {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            let _e3119_ = aux_scratch[_e3116_];
                                                            phi_4826_ = type_15_(0u, _e1495_);
                                                            phi_4829_ = _e3119_;
                                                            phi_4831_ = _e3115_;
                                                            loop {
                                                                let _e3122_ = phi_4826_;
                                                                let _e3124_ = phi_4829_;
                                                                let _e3126_ = phi_4831_;
                                                                local_19_ = _e3126_;
                                                                local_20_ = _e3124_;
                                                                if (_e3122_.member < _e3122_.member_1_) {
                                                                    phi_4827_ = type_15_((_e3122_.member + 1u), _e3122_.member_1_);
                                                                    phi_4854_ = type_15_(1u, _e3122_.member);
                                                                } else {
                                                                    phi_4827_ = _e3122_;
                                                                    phi_4854_ = type_15_(0u, u32());
                                                                }
                                                                let _e3139_ = phi_4827_;
                                                                let _e3141_ = phi_4854_;
                                                                let _e3145_ = (bitcast<i32>(_e3141_.member) != 0i);
                                                                if _e3145_ {
                                                                    let _e3146_ = (_e3141_.member_1_ & 127u);
                                                                    if (_e3146_ < 1536u) {
                                                                    } else {
                                                                        phi_35187_ = true;
                                                                        break;
                                                                    }
                                                                    let _e3149_ = scan_scratch[_e3146_];
                                                                    let _e3150_ = (_e2314_ * 0.5f);
                                                                    if (_e3150_ != _e3150_) {
                                                                        phi_24411_ = true;
                                                                    } else {
                                                                        phi_24411_ = (0.00001f >= _e3150_);
                                                                    }
                                                                    let _e3154_ = phi_24411_;
                                                                    let _e3155_ = select(_e3150_, 0.00001f, _e3154_);
                                                                    if (_e2272_ != _e2272_) {
                                                                        phi_24471_ = true;
                                                                    } else {
                                                                        phi_24471_ = (0f >= _e2272_);
                                                                    }
                                                                    let _e3159_ = phi_24471_;
                                                                    if (_e2321_ != _e2321_) {
                                                                        phi_24531_ = true;
                                                                    } else {
                                                                        phi_24531_ = (0f >= _e2321_);
                                                                    }
                                                                    let _e3164_ = phi_24531_;
                                                                    let _e3166_ = (_e3126_ * _e3126_);
                                                                    if (_e3166_ != _e3166_) {
                                                                        phi_24591_ = true;
                                                                    } else {
                                                                        phi_24591_ = (64f <= _e3166_);
                                                                    }
                                                                    let _e3170_ = phi_24591_;
                                                                    let _e3173_ = (select(_e2272_, 0f, _e3159_) + (select(_e2321_, 0f, _e3164_) * select(_e3166_, 64f, _e3170_)));
                                                                    let _e3176_ = (1f + (_e3155_ * (_e3155_ + _e3173_)));
                                                                    let _e3179_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3176_)));
                                                                    let _e3183_ = (_e3179_ + (_e3179_ * (1f - (_e3176_ * _e3179_))));
                                                                    let _e3187_ = (_e3183_ + (_e3183_ * (1f - (_e3176_ * _e3183_))));
                                                                    let _e3188_ = (_e3155_ * _e3187_);
                                                                    let _e3191_ = (_e3149_ - _e3124_);
                                                                    let _e3194_ = ((_e3187_ * _e3126_) + (_e3188_ * _e3191_));
                                                                    let _e3198_ = ((_e3124_ + (_e3188_ * _e3126_)) + ((_e3155_ * _e3188_) * _e3191_));
                                                                    let _e3201_ = bitcast<u32>(((_e3194_ + _e3194_) - _e3126_));
                                                                    let _e3202_ = (_e3201_ & 2139095040u);
                                                                    if (_e3202_ == 2139095040u) {
                                                                        phi_5021_ = 0u;
                                                                    } else {
                                                                        if (_e3202_ == 0u) {
                                                                            phi_5020_ = (_e3201_ & 2147483648u);
                                                                        } else {
                                                                            phi_5020_ = _e3201_;
                                                                        }
                                                                        let _e3207_ = phi_5020_;
                                                                        phi_5021_ = _e3207_;
                                                                    }
                                                                    let _e3209_ = phi_5021_;
                                                                    let _e3213_ = bitcast<u32>(((_e3198_ + _e3198_) - _e3124_));
                                                                    let _e3214_ = (_e3213_ & 2139095040u);
                                                                    if (_e3214_ == 2139095040u) {
                                                                        phi_5073_ = 0u;
                                                                    } else {
                                                                        if (_e3214_ == 0u) {
                                                                            phi_5072_ = (_e3213_ & 2147483648u);
                                                                        } else {
                                                                            phi_5072_ = _e3213_;
                                                                        }
                                                                        let _e3219_ = phi_5072_;
                                                                        phi_5073_ = _e3219_;
                                                                    }
                                                                    let _e3221_ = phi_5073_;
                                                                    let _e3239_ = (_e3146_ & 127u);
                                                                    let _e3240_ = ((_e2260_.member_1_ * 512u) + _e3239_);
                                                                    if (_e3240_ < 2048u) {
                                                                    } else {
                                                                        phi_35187_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3240_] = (((((_e3198_ * _e2279_) + (_e3194_ * _e2286_)) + (((_e3149_ - (_e3173_ * _e3194_)) - _e3198_) * _e2293_)) + ((_e3149_ - ((_e3173_ + _e3173_) * _e3194_)) * _e2300_)) + (_e3149_ * _e2307_));
                                                                    let _e3245_ = ((((_e2260_.member_1_ * 4u) + 1u) * 128u) + _e3239_);
                                                                    if (_e3245_ < 2048u) {
                                                                    } else {
                                                                        phi_35187_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3245_] = _e3194_;
                                                                    phi_4830_ = bitcast<f32>(_e3221_);
                                                                    phi_4832_ = bitcast<f32>(_e3209_);
                                                                } else {
                                                                    phi_4830_ = f32();
                                                                    phi_4832_ = f32();
                                                                }
                                                                let _e3249_ = phi_4830_;
                                                                let _e3251_ = phi_4832_;
                                                                phi_4826_ = _e3139_;
                                                                phi_4829_ = _e3249_;
                                                                phi_4831_ = _e3251_;
                                                                phi_35187_ = _e3102_;
                                                                if !(_e3145_) {
                                                                    break;
                                                                }
                                                            }
                                                            let _e3254_ = phi_35187_;
                                                            phi_35271_ = _e3254_;
                                                            if _e3254_ {
                                                                break;
                                                            }
                                                            if _e3113_ {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            let _e3256_ = local_19_;
                                                            aux_scratch[_e2260_.member_1_] = _e3256_;
                                                            if _e3117_ {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            let _e3258_ = local_20_;
                                                            aux_scratch[_e3116_] = _e3258_;
                                                            phi_35234_ = _e3254_;
                                                        } else {
                                                            phi_35234_ = _e3102_;
                                                        }
                                                        let _e3260_ = phi_35234_;
                                                        if (_e157_ == 1u) {
                                                            let _e3262_ = (8u + _e2260_.member_1_);
                                                            let _e3263_ = (_e3262_ < 256u);
                                                            if _e3263_ {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            let _e3265_ = aux_scratch[_e3262_];
                                                            let _e3266_ = (12u + _e2260_.member_1_);
                                                            let _e3267_ = (_e3266_ < 256u);
                                                            if _e3267_ {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            let _e3269_ = aux_scratch[_e3266_];
                                                            phi_5205_ = type_15_(0u, _e1495_);
                                                            phi_5208_ = _e3269_;
                                                            phi_5210_ = _e3265_;
                                                            loop {
                                                                let _e3272_ = phi_5205_;
                                                                let _e3274_ = phi_5208_;
                                                                let _e3276_ = phi_5210_;
                                                                local_21_ = _e3276_;
                                                                local_22_ = _e3274_;
                                                                if (_e3272_.member < _e3272_.member_1_) {
                                                                    phi_5206_ = type_15_((_e3272_.member + 1u), _e3272_.member_1_);
                                                                    phi_5233_ = type_15_(1u, _e3272_.member);
                                                                } else {
                                                                    phi_5206_ = _e3272_;
                                                                    phi_5233_ = type_15_(0u, u32());
                                                                }
                                                                let _e3289_ = phi_5206_;
                                                                let _e3291_ = phi_5233_;
                                                                let _e3295_ = (bitcast<i32>(_e3291_.member) != 0i);
                                                                if _e3295_ {
                                                                    let _e3296_ = (_e3291_.member_1_ & 127u);
                                                                    let _e3297_ = (128u + _e3296_);
                                                                    if (_e3297_ < 1536u) {
                                                                    } else {
                                                                        phi_35230_ = true;
                                                                        break;
                                                                    }
                                                                    let _e3300_ = scan_scratch[_e3297_];
                                                                    let _e3301_ = (_e2314_ * 0.5f);
                                                                    if (_e3301_ != _e3301_) {
                                                                        phi_24651_ = true;
                                                                    } else {
                                                                        phi_24651_ = (0.00001f >= _e3301_);
                                                                    }
                                                                    let _e3305_ = phi_24651_;
                                                                    let _e3306_ = select(_e3301_, 0.00001f, _e3305_);
                                                                    if (_e2272_ != _e2272_) {
                                                                        phi_24711_ = true;
                                                                    } else {
                                                                        phi_24711_ = (0f >= _e2272_);
                                                                    }
                                                                    let _e3310_ = phi_24711_;
                                                                    if (_e2321_ != _e2321_) {
                                                                        phi_24771_ = true;
                                                                    } else {
                                                                        phi_24771_ = (0f >= _e2321_);
                                                                    }
                                                                    let _e3315_ = phi_24771_;
                                                                    let _e3317_ = (_e3276_ * _e3276_);
                                                                    if (_e3317_ != _e3317_) {
                                                                        phi_24831_ = true;
                                                                    } else {
                                                                        phi_24831_ = (64f <= _e3317_);
                                                                    }
                                                                    let _e3321_ = phi_24831_;
                                                                    let _e3324_ = (select(_e2272_, 0f, _e3310_) + (select(_e2321_, 0f, _e3315_) * select(_e3317_, 64f, _e3321_)));
                                                                    let _e3327_ = (1f + (_e3306_ * (_e3306_ + _e3324_)));
                                                                    let _e3330_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3327_)));
                                                                    let _e3334_ = (_e3330_ + (_e3330_ * (1f - (_e3327_ * _e3330_))));
                                                                    let _e3338_ = (_e3334_ + (_e3334_ * (1f - (_e3327_ * _e3334_))));
                                                                    let _e3339_ = (_e3306_ * _e3338_);
                                                                    let _e3342_ = (_e3300_ - _e3274_);
                                                                    let _e3345_ = ((_e3338_ * _e3276_) + (_e3339_ * _e3342_));
                                                                    let _e3349_ = ((_e3274_ + (_e3339_ * _e3276_)) + ((_e3306_ * _e3339_) * _e3342_));
                                                                    let _e3352_ = bitcast<u32>(((_e3345_ + _e3345_) - _e3276_));
                                                                    let _e3353_ = (_e3352_ & 2139095040u);
                                                                    if (_e3353_ == 2139095040u) {
                                                                        phi_5401_ = 0u;
                                                                    } else {
                                                                        if (_e3353_ == 0u) {
                                                                            phi_5400_ = (_e3352_ & 2147483648u);
                                                                        } else {
                                                                            phi_5400_ = _e3352_;
                                                                        }
                                                                        let _e3358_ = phi_5400_;
                                                                        phi_5401_ = _e3358_;
                                                                    }
                                                                    let _e3360_ = phi_5401_;
                                                                    let _e3364_ = bitcast<u32>(((_e3349_ + _e3349_) - _e3274_));
                                                                    let _e3365_ = (_e3364_ & 2139095040u);
                                                                    if (_e3365_ == 2139095040u) {
                                                                        phi_5453_ = 0u;
                                                                    } else {
                                                                        if (_e3365_ == 0u) {
                                                                            phi_5452_ = (_e3364_ & 2147483648u);
                                                                        } else {
                                                                            phi_5452_ = _e3364_;
                                                                        }
                                                                        let _e3370_ = phi_5452_;
                                                                        phi_5453_ = _e3370_;
                                                                    }
                                                                    let _e3372_ = phi_5453_;
                                                                    let _e3388_ = (_e2260_.member_1_ * 4u);
                                                                    let _e3391_ = (_e3296_ & 127u);
                                                                    let _e3392_ = (((_e3388_ + 2u) * 128u) + _e3391_);
                                                                    if (_e3392_ < 2048u) {
                                                                    } else {
                                                                        phi_35230_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3392_] = (((((_e3349_ * _e2279_) + (_e3345_ * _e2286_)) + (((_e3300_ - (_e3324_ * _e3345_)) - _e3349_) * _e2293_)) + ((_e3300_ - ((_e3324_ + _e3324_) * _e3345_)) * _e2300_)) + (_e3300_ * _e2307_));
                                                                    let _e3397_ = (((_e3388_ + 3u) * 128u) + _e3391_);
                                                                    if (_e3397_ < 2048u) {
                                                                    } else {
                                                                        phi_35230_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3397_] = _e3345_;
                                                                    phi_5209_ = bitcast<f32>(_e3372_);
                                                                    phi_5211_ = bitcast<f32>(_e3360_);
                                                                } else {
                                                                    phi_5209_ = f32();
                                                                    phi_5211_ = f32();
                                                                }
                                                                let _e3401_ = phi_5209_;
                                                                let _e3403_ = phi_5211_;
                                                                phi_5205_ = _e3289_;
                                                                phi_5208_ = _e3401_;
                                                                phi_5210_ = _e3403_;
                                                                phi_35230_ = _e3260_;
                                                                if !(_e3295_) {
                                                                    break;
                                                                }
                                                            }
                                                            let _e3406_ = phi_35230_;
                                                            phi_35271_ = _e3406_;
                                                            if _e3406_ {
                                                                break;
                                                            }
                                                            if _e3263_ {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            let _e3408_ = local_21_;
                                                            aux_scratch[_e3262_] = _e3408_;
                                                            if _e3267_ {
                                                            } else {
                                                                phi_35271_ = true;
                                                                break;
                                                            }
                                                            let _e3410_ = local_22_;
                                                            aux_scratch[_e3266_] = _e3410_;
                                                            phi_35274_ = _e3406_;
                                                        } else {
                                                            phi_35274_ = _e3260_;
                                                        }
                                                        let _e3412_ = phi_35274_;
                                                        workgroupBarrier();
                                                        phi_35273_ = _e3412_;
                                                    }
                                                    let _e3527_ = phi_35273_;
                                                    phi_35272_ = _e3527_;
                                                } else {
                                                    phi_35272_ = _e2243_;
                                                }
                                                let _e3529_ = phi_35272_;
                                                phi_35084_ = _e3529_;
                                                phi_2777_ = _e2258_;
                                                phi_35271_ = _e3529_;
                                                if !(_e2264_) {
                                                    break;
                                                }
                                            }
                                            let _e3532_ = phi_35271_;
                                            phi_37452_ = _e3532_;
                                            if _e3532_ {
                                                break;
                                            }
                                            phi_35312_ = _e3532_;
                                        } else {
                                            if (_e157_ < 128u) {
                                                let _e1933_ = vec4(_e1497_);
                                                let _e1934_ = select(vec4<f32>(0f, 0f, 0f, 0f), _e1518_, _e1933_);
                                                let _e1935_ = select(vec4<f32>(0f, 0f, 0f, 0f), _e1516_, _e1933_);
                                                phi_5578_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e1937_ = phi_5578_;
                                                    if (_e1937_.member < _e1937_.member_1_) {
                                                        phi_5579_ = type_15_((_e1937_.member + 1u), _e1937_.member_1_);
                                                        phi_5602_ = type_15_(1u, _e1937_.member);
                                                    } else {
                                                        phi_5579_ = _e1937_;
                                                        phi_5602_ = type_15_(0u, u32());
                                                    }
                                                    let _e1950_ = phi_5579_;
                                                    let _e1952_ = phi_5602_;
                                                    let _e1956_ = (bitcast<i32>(_e1952_.member) != 0i);
                                                    if _e1956_ {
                                                        let _e1957_ = (_e1952_.member_1_ & 3u);
                                                        switch bitcast<i32>(_e1957_) {
                                                            case 0: {
                                                                phi_5620_ = _e1934_.x;
                                                                break;
                                                            }
                                                            case 1: {
                                                                phi_5620_ = _e1934_.y;
                                                                break;
                                                            }
                                                            case 2: {
                                                                phi_5620_ = _e1934_.z;
                                                                break;
                                                            }
                                                            default: {
                                                                phi_5620_ = _e1934_.w;
                                                                break;
                                                            }
                                                        }
                                                        let _e1964_ = phi_5620_;
                                                        let _e1965_ = bitcast<u32>(_e1964_);
                                                        let _e1966_ = (_e1965_ & 2139095040u);
                                                        if (_e1966_ == 2139095040u) {
                                                            phi_5631_ = true;
                                                        } else {
                                                            phi_5631_ = (_e1966_ == 0u);
                                                        }
                                                        let _e1970_ = phi_5631_;
                                                        let _e1975_ = ((_e1952_.member_1_ * 256u) + _e158_);
                                                        if (_e1975_ < 1536u) {
                                                        } else {
                                                            phi_34994_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1975_] = bitcast<f32>(select(_e1965_, 0u, _e1970_));
                                                        switch bitcast<i32>(_e1957_) {
                                                            case 0: {
                                                                phi_5656_ = _e1935_.x;
                                                                break;
                                                            }
                                                            case 1: {
                                                                phi_5656_ = _e1935_.y;
                                                                break;
                                                            }
                                                            case 2: {
                                                                phi_5656_ = _e1935_.z;
                                                                break;
                                                            }
                                                            default: {
                                                                phi_5656_ = _e1935_.w;
                                                                break;
                                                            }
                                                        }
                                                        let _e1984_ = phi_5656_;
                                                        let _e1985_ = bitcast<u32>(_e1984_);
                                                        let _e1986_ = (_e1985_ & 2139095040u);
                                                        if (_e1986_ == 2139095040u) {
                                                            phi_5667_ = true;
                                                        } else {
                                                            phi_5667_ = (_e1986_ == 0u);
                                                        }
                                                        let _e1990_ = phi_5667_;
                                                        let _e1995_ = ((((_e1952_.member_1_ * 2u) + 1u) * 128u) + _e158_);
                                                        if (_e1995_ < 1536u) {
                                                        } else {
                                                            phi_34994_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1995_] = bitcast<f32>(select(_e1985_, 0u, _e1990_));
                                                    }
                                                    phi_5578_ = _e1950_;
                                                    phi_34994_ = _e1466_;
                                                    if !(_e1956_) {
                                                        break;
                                                    }
                                                }
                                                let _e2000_ = phi_34994_;
                                                phi_37452_ = _e2000_;
                                                if _e2000_ {
                                                    break;
                                                }
                                                phi_35021_ = _e2000_;
                                            } else {
                                                phi_35021_ = _e1466_;
                                            }
                                            let _e2002_ = phi_35021_;
                                            workgroupBarrier();
                                            if (_e157_ < 8u) {
                                                let _e2005_ = (_e157_ >> bitcast<u32>(1i));
                                                let _e2006_ = (_e157_ & 1u);
                                                let _e2007_ = (_e2006_ != 0u);
                                                let _e2008_ = (_e2005_ & 3u);
                                                switch bitcast<i32>(_e2008_) {
                                                    case 0: {
                                                        phi_5795_ = _e321_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5795_ = _e321_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5795_ = _e321_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5795_ = _e321_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2015_ = phi_5795_;
                                                switch bitcast<i32>(_e2008_) {
                                                    case 0: {
                                                        phi_5812_ = _e335_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5812_ = _e335_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5812_ = _e335_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5812_ = _e335_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2022_ = phi_5812_;
                                                switch bitcast<i32>(_e2008_) {
                                                    case 0: {
                                                        phi_5823_ = _e342_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5823_ = _e342_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5823_ = _e342_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5823_ = _e342_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2029_ = phi_5823_;
                                                switch bitcast<i32>(_e2008_) {
                                                    case 0: {
                                                        phi_5834_ = _e349_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5834_ = _e349_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5834_ = _e349_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5834_ = _e349_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2036_ = phi_5834_;
                                                switch bitcast<i32>(_e2008_) {
                                                    case 0: {
                                                        phi_5845_ = _e356_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5845_ = _e356_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5845_ = _e356_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5845_ = _e356_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2043_ = phi_5845_;
                                                switch bitcast<i32>(_e2008_) {
                                                    case 0: {
                                                        phi_5856_ = _e363_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5856_ = _e363_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5856_ = _e363_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5856_ = _e363_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2050_ = phi_5856_;
                                                switch bitcast<i32>(_e2008_) {
                                                    case 0: {
                                                        phi_6071_ = _e412_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6071_ = _e412_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6071_ = _e412_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6071_ = _e412_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2057_ = phi_6071_;
                                                switch bitcast<i32>(_e2008_) {
                                                    case 0: {
                                                        phi_6146_ = _e463_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6146_ = _e463_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6146_ = _e463_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6146_ = _e463_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2064_ = phi_6146_;
                                                if _e2007_ {
                                                    phi_6151_ = (8u + _e2005_);
                                                } else {
                                                    phi_6151_ = _e2005_;
                                                }
                                                let _e2067_ = phi_6151_;
                                                if _e2007_ {
                                                    phi_6157_ = (12u + _e2005_);
                                                } else {
                                                    phi_6157_ = (4u + _e2005_);
                                                }
                                                let _e2071_ = phi_6157_;
                                                let _e2072_ = (_e2067_ < 256u);
                                                if _e2072_ {
                                                } else {
                                                    phi_37452_ = true;
                                                    break;
                                                }
                                                let _e2074_ = aux_scratch[_e2067_];
                                                let _e2075_ = (_e2071_ < 256u);
                                                if _e2075_ {
                                                } else {
                                                    phi_37452_ = true;
                                                    break;
                                                }
                                                let _e2077_ = aux_scratch[_e2071_];
                                                let _e2081_ = select(0u, 2u, _e2007_);
                                                phi_6182_ = type_15_(0u, _e1495_);
                                                phi_6185_ = _e2077_;
                                                phi_6187_ = _e2074_;
                                                loop {
                                                    let _e2084_ = phi_6182_;
                                                    let _e2086_ = phi_6185_;
                                                    let _e2088_ = phi_6187_;
                                                    local_16_ = _e2088_;
                                                    local_17_ = _e2086_;
                                                    if (_e2084_.member < _e2084_.member_1_) {
                                                        phi_6183_ = type_15_((_e2084_.member + 1u), _e2084_.member_1_);
                                                        phi_6210_ = type_15_(1u, _e2084_.member);
                                                    } else {
                                                        phi_6183_ = _e2084_;
                                                        phi_6210_ = type_15_(0u, u32());
                                                    }
                                                    let _e2101_ = phi_6183_;
                                                    let _e2103_ = phi_6210_;
                                                    let _e2107_ = (bitcast<i32>(_e2103_.member) != 0i);
                                                    if _e2107_ {
                                                        let _e2108_ = (_e2103_.member_1_ & 127u);
                                                        let _e2109_ = ((((_e2005_ * 2u) + _e2006_) * 128u) + _e2108_);
                                                        if (_e2109_ < 1536u) {
                                                        } else {
                                                            phi_35006_ = true;
                                                            break;
                                                        }
                                                        let _e2112_ = scan_scratch[_e2109_];
                                                        let _e2113_ = (_e2057_ * 0.5f);
                                                        if (_e2113_ != _e2113_) {
                                                            phi_23631_ = true;
                                                        } else {
                                                            phi_23631_ = (0.00001f >= _e2113_);
                                                        }
                                                        let _e2117_ = phi_23631_;
                                                        let _e2118_ = select(_e2113_, 0.00001f, _e2117_);
                                                        if (_e2015_ != _e2015_) {
                                                            phi_23691_ = true;
                                                        } else {
                                                            phi_23691_ = (0f >= _e2015_);
                                                        }
                                                        let _e2122_ = phi_23691_;
                                                        if (_e2064_ != _e2064_) {
                                                            phi_23751_ = true;
                                                        } else {
                                                            phi_23751_ = (0f >= _e2064_);
                                                        }
                                                        let _e2127_ = phi_23751_;
                                                        let _e2129_ = (_e2088_ * _e2088_);
                                                        if (_e2129_ != _e2129_) {
                                                            phi_23811_ = true;
                                                        } else {
                                                            phi_23811_ = (64f <= _e2129_);
                                                        }
                                                        let _e2133_ = phi_23811_;
                                                        let _e2136_ = (select(_e2015_, 0f, _e2122_) + (select(_e2064_, 0f, _e2127_) * select(_e2129_, 64f, _e2133_)));
                                                        let _e2139_ = (1f + (_e2118_ * (_e2118_ + _e2136_)));
                                                        let _e2142_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2139_)));
                                                        let _e2146_ = (_e2142_ + (_e2142_ * (1f - (_e2139_ * _e2142_))));
                                                        let _e2150_ = (_e2146_ + (_e2146_ * (1f - (_e2139_ * _e2146_))));
                                                        let _e2151_ = (_e2118_ * _e2150_);
                                                        let _e2154_ = (_e2112_ - _e2086_);
                                                        let _e2157_ = ((_e2150_ * _e2088_) + (_e2151_ * _e2154_));
                                                        let _e2161_ = ((_e2086_ + (_e2151_ * _e2088_)) + ((_e2118_ * _e2151_) * _e2154_));
                                                        let _e2164_ = bitcast<u32>(((_e2157_ + _e2157_) - _e2088_));
                                                        let _e2165_ = (_e2164_ & 2139095040u);
                                                        if (_e2165_ == 2139095040u) {
                                                            phi_6378_ = 0u;
                                                        } else {
                                                            if (_e2165_ == 0u) {
                                                                phi_6377_ = (_e2164_ & 2147483648u);
                                                            } else {
                                                                phi_6377_ = _e2164_;
                                                            }
                                                            let _e2170_ = phi_6377_;
                                                            phi_6378_ = _e2170_;
                                                        }
                                                        let _e2172_ = phi_6378_;
                                                        let _e2176_ = bitcast<u32>(((_e2161_ + _e2161_) - _e2086_));
                                                        let _e2177_ = (_e2176_ & 2139095040u);
                                                        if (_e2177_ == 2139095040u) {
                                                            phi_6430_ = 0u;
                                                        } else {
                                                            if (_e2177_ == 0u) {
                                                                phi_6429_ = (_e2176_ & 2147483648u);
                                                            } else {
                                                                phi_6429_ = _e2176_;
                                                            }
                                                            let _e2182_ = phi_6429_;
                                                            phi_6430_ = _e2182_;
                                                        }
                                                        let _e2184_ = phi_6430_;
                                                        let _e2200_ = (_e2005_ * 4u);
                                                        let _e2203_ = (_e2108_ & 127u);
                                                        let _e2204_ = (((_e2200_ + _e2081_) * 128u) + _e2203_);
                                                        if (_e2204_ < 2048u) {
                                                        } else {
                                                            phi_35006_ = true;
                                                            break;
                                                        }
                                                        lane_buf[_e2204_] = (((((_e2161_ * _e2022_) + (_e2157_ * _e2029_)) + (((_e2112_ - (_e2136_ * _e2157_)) - _e2161_) * _e2036_)) + ((_e2112_ - ((_e2136_ + _e2136_) * _e2157_)) * _e2043_)) + (_e2112_ * _e2050_));
                                                        let _e2210_ = (((_e2200_ + (_e2081_ + 1u)) * 128u) + _e2203_);
                                                        if (_e2210_ < 2048u) {
                                                        } else {
                                                            phi_35006_ = true;
                                                            break;
                                                        }
                                                        lane_buf[_e2210_] = _e2157_;
                                                        phi_6186_ = bitcast<f32>(_e2184_);
                                                        phi_6188_ = bitcast<f32>(_e2172_);
                                                    } else {
                                                        phi_6186_ = f32();
                                                        phi_6188_ = f32();
                                                    }
                                                    let _e2214_ = phi_6186_;
                                                    let _e2216_ = phi_6188_;
                                                    phi_6182_ = _e2101_;
                                                    phi_6185_ = _e2214_;
                                                    phi_6187_ = _e2216_;
                                                    phi_35006_ = _e2002_;
                                                    if !(_e2107_) {
                                                        break;
                                                    }
                                                }
                                                let _e2219_ = phi_35006_;
                                                phi_37452_ = _e2219_;
                                                if _e2219_ {
                                                    break;
                                                }
                                                let _e2221_ = local_16_;
                                                let _e2222_ = bitcast<u32>(_e2221_);
                                                let _e2223_ = (_e2222_ & 2139095040u);
                                                if (_e2223_ == 2139095040u) {
                                                    phi_6538_ = true;
                                                } else {
                                                    phi_6538_ = (_e2223_ == 0u);
                                                }
                                                let _e2227_ = phi_6538_;
                                                if _e2072_ {
                                                } else {
                                                    phi_37452_ = true;
                                                    break;
                                                }
                                                aux_scratch[_e2067_] = bitcast<f32>(select(_e2222_, 0u, _e2227_));
                                                let _e2231_ = local_17_;
                                                let _e2232_ = bitcast<u32>(_e2231_);
                                                let _e2233_ = (_e2232_ & 2139095040u);
                                                if (_e2233_ == 2139095040u) {
                                                    phi_6560_ = true;
                                                } else {
                                                    phi_6560_ = (_e2233_ == 0u);
                                                }
                                                let _e2237_ = phi_6560_;
                                                if _e2075_ {
                                                } else {
                                                    phi_37452_ = true;
                                                    break;
                                                }
                                                aux_scratch[_e2071_] = bitcast<f32>(select(_e2232_, 0u, _e2237_));
                                                phi_35313_ = _e2219_;
                                            } else {
                                                phi_35313_ = _e2002_;
                                            }
                                            let _e2241_ = phi_35313_;
                                            workgroupBarrier();
                                            phi_35312_ = _e2241_;
                                        }
                                        let _e3534_ = phi_35312_;
                                        phi_35311_ = _e3534_;
                                    } else {
                                        phi_34883_ = _e1466_;
                                        phi_6574_ = type_15_(0u, 4u);
                                        loop {
                                            let _e1520_ = phi_34883_;
                                            let _e1522_ = phi_6574_;
                                            if (_e1522_.member < _e1522_.member_1_) {
                                                phi_6575_ = type_15_((_e1522_.member + 1u), _e1522_.member_1_);
                                                phi_6598_ = type_15_(1u, _e1522_.member);
                                            } else {
                                                phi_6575_ = _e1522_;
                                                phi_6598_ = type_15_(0u, u32());
                                            }
                                            let _e1535_ = phi_6575_;
                                            let _e1537_ = phi_6598_;
                                            let _e1541_ = (bitcast<i32>(_e1537_.member) != 0i);
                                            if _e1541_ {
                                                let _e1542_ = (_e1537_.member_1_ & 3u);
                                                switch bitcast<i32>(_e1542_) {
                                                    case 0: {
                                                        phi_6693_ = _e300_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6693_ = _e300_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6693_ = _e300_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6693_ = _e300_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1549_ = phi_6693_;
                                                switch bitcast<i32>(_e1542_) {
                                                    case 0: {
                                                        phi_6704_ = _e307_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6704_ = _e307_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6704_ = _e307_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6704_ = _e307_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1556_ = phi_6704_;
                                                switch bitcast<i32>(_e1542_) {
                                                    case 0: {
                                                        phi_6715_ = _e314_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6715_ = _e314_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6715_ = _e314_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6715_ = _e314_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1563_ = phi_6715_;
                                                switch bitcast<i32>(_e1542_) {
                                                    case 0: {
                                                        phi_6726_ = _e321_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6726_ = _e321_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6726_ = _e321_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6726_ = _e321_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1570_ = phi_6726_;
                                                switch bitcast<i32>(_e1542_) {
                                                    case 0: {
                                                        phi_6737_ = _e328_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6737_ = _e328_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6737_ = _e328_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6737_ = _e328_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1577_ = phi_6737_;
                                                switch bitcast<i32>(_e1542_) {
                                                    case 0: {
                                                        phi_6748_ = _e335_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6748_ = _e335_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6748_ = _e335_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6748_ = _e335_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1584_ = phi_6748_;
                                                switch bitcast<i32>(_e1542_) {
                                                    case 0: {
                                                        phi_6759_ = _e342_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6759_ = _e342_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6759_ = _e342_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6759_ = _e342_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1591_ = phi_6759_;
                                                switch bitcast<i32>(_e1542_) {
                                                    case 0: {
                                                        phi_6770_ = _e349_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6770_ = _e349_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6770_ = _e349_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6770_ = _e349_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1598_ = phi_6770_;
                                                switch bitcast<i32>(_e1542_) {
                                                    case 0: {
                                                        phi_6781_ = _e356_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6781_ = _e356_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6781_ = _e356_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6781_ = _e356_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1605_ = phi_6781_;
                                                switch bitcast<i32>(_e1542_) {
                                                    case 0: {
                                                        phi_6792_ = _e363_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6792_ = _e363_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6792_ = _e363_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6792_ = _e363_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1612_ = phi_6792_;
                                                if _e1497_ {
                                                    let _e1614_ = select(_e1518_, _e1516_, vec4(_e161_));
                                                    switch bitcast<i32>(_e1542_) {
                                                        case 0: {
                                                            phi_7100_ = _e1614_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_7100_ = _e1614_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_7100_ = _e1614_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_7100_ = _e1614_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e1621_ = phi_7100_;
                                                    let _e1622_ = bitcast<u32>(_e1621_);
                                                    let _e1623_ = (_e1622_ & 2139095040u);
                                                    if (_e1623_ == 2139095040u) {
                                                        phi_7111_ = true;
                                                    } else {
                                                        phi_7111_ = (_e1623_ == 0u);
                                                    }
                                                    let _e1627_ = phi_7111_;
                                                    phi_7117_ = bitcast<f32>(select(_e1622_, 0u, _e1627_));
                                                } else {
                                                    phi_7117_ = 0f;
                                                }
                                                let _e1631_ = phi_7117_;
                                                if _e1497_ {
                                                    let _e1632_ = (_e1556_ + _e1556_);
                                                    let _e1633_ = (_e1563_ + _e1563_);
                                                    phi_7131_ = type_31_(((_e1549_ + _e1549_) - 1f), -(_e1632_), _e1632_, (1f - _e1633_), (_e1632_ * _e1631_), (_e1633_ * _e1631_));
                                                } else {
                                                    phi_7131_ = type_31_(1f, 0f, 0f, 1f, 0f, 0f);
                                                }
                                                let _e1642_ = phi_7131_;
                                                if _e161_ {
                                                    phi_7136_ = (8u + _e1537_.member_1_);
                                                } else {
                                                    phi_7136_ = _e1537_.member_1_;
                                                }
                                                let _e1645_ = phi_7136_;
                                                if _e161_ {
                                                    phi_7142_ = (12u + _e1537_.member_1_);
                                                } else {
                                                    phi_7142_ = (4u + _e1537_.member_1_);
                                                }
                                                let _e1649_ = phi_7142_;
                                                if (_e1645_ < 256u) {
                                                } else {
                                                    phi_34977_ = true;
                                                    break;
                                                }
                                                let _e1652_ = aux_scratch[_e1645_];
                                                if (_e1649_ < 256u) {
                                                } else {
                                                    phi_34977_ = true;
                                                    break;
                                                }
                                                let _e1655_ = aux_scratch[_e1649_];
                                                let _e1656_ = array<f32, 2>(_e1652_, _e1655_);
                                                let _e1657_ = select(0u, 768u, _e161_);
                                                let _e1658_ = (_e158_ & 127u);
                                                let _e1660_ = (_e1657_ + _e1658_);
                                                let _e1661_ = (_e1660_ < 1536u);
                                                if _e1661_ {
                                                } else {
                                                    phi_34977_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e1660_] = _e1642_.member;
                                                let _e1664_ = (_e1657_ + 128u);
                                                let _e1665_ = (_e1664_ + _e1658_);
                                                let _e1666_ = (_e1665_ < 1536u);
                                                if _e1666_ {
                                                } else {
                                                    phi_34977_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e1665_] = _e1642_.member_1_;
                                                let _e1669_ = (_e1657_ + 256u);
                                                let _e1670_ = (_e1669_ + _e1658_);
                                                let _e1671_ = (_e1670_ < 1536u);
                                                if _e1671_ {
                                                } else {
                                                    phi_34977_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e1670_] = _e1642_.member_2_;
                                                let _e1674_ = (_e1657_ + 384u);
                                                let _e1675_ = (_e1674_ + _e1658_);
                                                let _e1676_ = (_e1675_ < 1536u);
                                                if _e1676_ {
                                                } else {
                                                    phi_34977_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e1675_] = _e1642_.member_3_;
                                                let _e1679_ = (_e1657_ + 512u);
                                                let _e1680_ = (_e1679_ + _e1658_);
                                                let _e1681_ = (_e1680_ < 1536u);
                                                if _e1681_ {
                                                } else {
                                                    phi_34977_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e1680_] = _e1642_.member_4_;
                                                let _e1684_ = (_e1657_ + 640u);
                                                let _e1685_ = (_e1684_ + _e1658_);
                                                let _e1686_ = (_e1685_ < 1536u);
                                                if _e1686_ {
                                                } else {
                                                    phi_34977_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e1685_] = _e1642_.member_5_;
                                                workgroupBarrier();
                                                phi_7220_ = 1u;
                                                phi_7223_ = _e1642_;
                                                loop {
                                                    let _e1689_ = phi_7220_;
                                                    let _e1691_ = phi_7223_;
                                                    let _e1692_ = (_e1689_ < 128u);
                                                    if _e1692_ {
                                                        if (_e158_ >= _e1689_) {
                                                            let _e1695_ = ((_e158_ - _e1689_) & 127u);
                                                            let _e1696_ = (_e1657_ + _e1695_);
                                                            if (_e1696_ < 1536u) {
                                                            } else {
                                                                phi_34864_ = true;
                                                                break;
                                                            }
                                                            let _e1699_ = scan_scratch[_e1696_];
                                                            let _e1700_ = (_e1664_ + _e1695_);
                                                            if (_e1700_ < 1536u) {
                                                            } else {
                                                                phi_34864_ = true;
                                                                break;
                                                            }
                                                            let _e1703_ = scan_scratch[_e1700_];
                                                            let _e1704_ = (_e1669_ + _e1695_);
                                                            if (_e1704_ < 1536u) {
                                                            } else {
                                                                phi_34864_ = true;
                                                                break;
                                                            }
                                                            let _e1707_ = scan_scratch[_e1704_];
                                                            let _e1708_ = (_e1674_ + _e1695_);
                                                            if (_e1708_ < 1536u) {
                                                            } else {
                                                                phi_34864_ = true;
                                                                break;
                                                            }
                                                            let _e1711_ = scan_scratch[_e1708_];
                                                            let _e1712_ = (_e1679_ + _e1695_);
                                                            if (_e1712_ < 1536u) {
                                                            } else {
                                                                phi_34864_ = true;
                                                                break;
                                                            }
                                                            let _e1715_ = scan_scratch[_e1712_];
                                                            let _e1716_ = (_e1684_ + _e1695_);
                                                            if (_e1716_ < 1536u) {
                                                            } else {
                                                                phi_34864_ = true;
                                                                break;
                                                            }
                                                            let _e1719_ = scan_scratch[_e1716_];
                                                            phi_7325_ = type_31_(((_e1691_.member * _e1699_) + (_e1691_.member_1_ * _e1707_)), ((_e1691_.member * _e1703_) + (_e1691_.member_1_ * _e1711_)), ((_e1691_.member_2_ * _e1699_) + (_e1691_.member_3_ * _e1707_)), ((_e1691_.member_2_ * _e1703_) + (_e1691_.member_3_ * _e1711_)), (((_e1691_.member * _e1715_) + (_e1691_.member_1_ * _e1719_)) + _e1691_.member_4_), (((_e1691_.member_2_ * _e1715_) + (_e1691_.member_3_ * _e1719_)) + _e1691_.member_5_));
                                                        } else {
                                                            phi_7325_ = _e1691_;
                                                        }
                                                        let _e1748_ = phi_7325_;
                                                        workgroupBarrier();
                                                        if _e1661_ {
                                                        } else {
                                                            phi_34864_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1660_] = _e1748_.member;
                                                        if _e1666_ {
                                                        } else {
                                                            phi_34864_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1665_] = _e1748_.member_1_;
                                                        if _e1671_ {
                                                        } else {
                                                            phi_34864_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1670_] = _e1748_.member_2_;
                                                        if _e1676_ {
                                                        } else {
                                                            phi_34864_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1675_] = _e1748_.member_3_;
                                                        if _e1681_ {
                                                        } else {
                                                            phi_34864_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1680_] = _e1748_.member_4_;
                                                        if _e1686_ {
                                                        } else {
                                                            phi_34864_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1685_] = _e1748_.member_5_;
                                                        workgroupBarrier();
                                                        phi_7221_ = (_e1689_ << bitcast<u32>(1i));
                                                        phi_7224_ = _e1748_;
                                                    } else {
                                                        phi_7221_ = u32();
                                                        phi_7224_ = type_31_();
                                                    }
                                                    let _e1758_ = phi_7221_;
                                                    let _e1760_ = phi_7224_;
                                                    phi_7220_ = _e1758_;
                                                    phi_7223_ = _e1760_;
                                                    phi_34864_ = _e1520_;
                                                    if !(_e1692_) {
                                                        break;
                                                    }
                                                }
                                                let _e1763_ = phi_34864_;
                                                phi_34977_ = _e1763_;
                                                if _e1763_ {
                                                    break;
                                                }
                                                if (_e158_ == 0u) {
                                                    phi_7449_ = _e1656_;
                                                } else {
                                                    let _e1766_ = ((_e158_ - 1u) & 127u);
                                                    let _e1767_ = (_e1657_ + _e1766_);
                                                    if (_e1767_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1770_ = scan_scratch[_e1767_];
                                                    let _e1771_ = (_e1664_ + _e1766_);
                                                    if (_e1771_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1774_ = scan_scratch[_e1771_];
                                                    let _e1775_ = (_e1669_ + _e1766_);
                                                    if (_e1775_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1778_ = scan_scratch[_e1775_];
                                                    let _e1779_ = (_e1674_ + _e1766_);
                                                    if (_e1779_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1782_ = scan_scratch[_e1779_];
                                                    let _e1783_ = (_e1679_ + _e1766_);
                                                    if (_e1783_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1786_ = scan_scratch[_e1783_];
                                                    let _e1787_ = (_e1684_ + _e1766_);
                                                    if (_e1787_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1790_ = scan_scratch[_e1787_];
                                                    phi_7449_ = array<f32, 2>((((_e1770_ * _e1652_) + (_e1774_ * _e1655_)) + _e1786_), (((_e1778_ * _e1652_) + (_e1782_ * _e1655_)) + _e1790_));
                                                }
                                                let _e1801_ = phi_7449_;
                                                if _e1496_ {
                                                    if (_e1495_ <= 128u) {
                                                        phi_7461_ = (_e1495_ - 1u);
                                                    } else {
                                                        phi_7461_ = 127u;
                                                    }
                                                    let _e1807_ = phi_7461_;
                                                    let _e1808_ = (_e1807_ & 127u);
                                                    let _e1809_ = (_e1657_ + _e1808_);
                                                    if (_e1809_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1812_ = scan_scratch[_e1809_];
                                                    let _e1813_ = (_e1664_ + _e1808_);
                                                    if (_e1813_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1816_ = scan_scratch[_e1813_];
                                                    let _e1817_ = (_e1669_ + _e1808_);
                                                    if (_e1817_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1820_ = scan_scratch[_e1817_];
                                                    let _e1821_ = (_e1674_ + _e1808_);
                                                    if (_e1821_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1824_ = scan_scratch[_e1821_];
                                                    let _e1825_ = (_e1679_ + _e1808_);
                                                    if (_e1825_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1828_ = scan_scratch[_e1825_];
                                                    let _e1829_ = (_e1684_ + _e1808_);
                                                    if (_e1829_ < 1536u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    let _e1832_ = scan_scratch[_e1829_];
                                                    phi_7525_ = array<f32, 2>((((_e1812_ * _e1652_) + (_e1816_ * _e1655_)) + _e1828_), (((_e1820_ * _e1652_) + (_e1824_ * _e1655_)) + _e1832_));
                                                } else {
                                                    phi_7525_ = _e1656_;
                                                }
                                                let _e1843_ = phi_7525_;
                                                workgroupBarrier();
                                                if _e1497_ {
                                                    let _e1844_ = (_e1631_ - _e1801_[1]);
                                                    let _e1847_ = ((_e1549_ * _e1801_[0]) + (_e1556_ * _e1844_));
                                                    let _e1851_ = ((_e1801_[1] + (_e1556_ * _e1801_[0])) + (_e1563_ * _e1844_));
                                                    let _e1866_ = select(0u, 2u, _e161_);
                                                    let _e1867_ = (_e1537_.member_1_ * 4u);
                                                    let _e1870_ = (((_e1867_ + _e1866_) * 128u) + _e1658_);
                                                    if (_e1870_ < 2048u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e1870_] = (((((_e1851_ * _e1584_) + (_e1847_ * _e1591_)) + (((_e1631_ - (_e1570_ * _e1847_)) - _e1851_) * _e1598_)) + ((_e1631_ - (_e1577_ * _e1847_)) * _e1605_)) + (_e1631_ * _e1612_));
                                                    let _e1876_ = (((_e1867_ + (_e1866_ + 1u)) * 128u) + _e1658_);
                                                    if (_e1876_ < 2048u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e1876_] = _e1847_;
                                                }
                                                if _e1098_ {
                                                    let _e1880_ = bitcast<u32>(_e1843_[0]);
                                                    let _e1881_ = (_e1880_ & 2139095040u);
                                                    if (_e1881_ == 2139095040u) {
                                                        phi_7780_ = true;
                                                    } else {
                                                        phi_7780_ = (_e1881_ == 0u);
                                                    }
                                                    let _e1885_ = phi_7780_;
                                                    if (_e1537_.member_1_ < 256u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1537_.member_1_] = bitcast<f32>(select(_e1880_, 0u, _e1885_));
                                                    let _e1891_ = bitcast<u32>(_e1843_[1]);
                                                    let _e1892_ = (_e1891_ & 2139095040u);
                                                    if (_e1892_ == 2139095040u) {
                                                        phi_7803_ = true;
                                                    } else {
                                                        phi_7803_ = (_e1892_ == 0u);
                                                    }
                                                    let _e1896_ = phi_7803_;
                                                    let _e1899_ = (4u + _e1537_.member_1_);
                                                    if (_e1899_ < 256u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1899_] = bitcast<f32>(select(_e1891_, 0u, _e1896_));
                                                }
                                                if (_e157_ == 128u) {
                                                    let _e1904_ = bitcast<u32>(_e1843_[0]);
                                                    let _e1905_ = (_e1904_ & 2139095040u);
                                                    if (_e1905_ == 2139095040u) {
                                                        phi_7831_ = true;
                                                    } else {
                                                        phi_7831_ = (_e1905_ == 0u);
                                                    }
                                                    let _e1909_ = phi_7831_;
                                                    let _e1912_ = (8u + _e1537_.member_1_);
                                                    if (_e1912_ < 256u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1912_] = bitcast<f32>(select(_e1904_, 0u, _e1909_));
                                                    let _e1916_ = bitcast<u32>(_e1843_[1]);
                                                    let _e1917_ = (_e1916_ & 2139095040u);
                                                    if (_e1917_ == 2139095040u) {
                                                        phi_7855_ = true;
                                                    } else {
                                                        phi_7855_ = (_e1917_ == 0u);
                                                    }
                                                    let _e1921_ = phi_7855_;
                                                    let _e1924_ = (12u + _e1537_.member_1_);
                                                    if (_e1924_ < 256u) {
                                                    } else {
                                                        phi_34977_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1924_] = bitcast<f32>(select(_e1916_, 0u, _e1921_));
                                                }
                                                workgroupBarrier();
                                                phi_34978_ = _e1763_;
                                            } else {
                                                phi_34978_ = _e1520_;
                                            }
                                            let _e1928_ = phi_34978_;
                                            phi_34883_ = _e1928_;
                                            phi_6574_ = _e1535_;
                                            phi_34977_ = _e1928_;
                                            if !(_e1541_) {
                                                break;
                                            }
                                        }
                                        let _e1931_ = phi_34977_;
                                        phi_37452_ = _e1931_;
                                        if _e1931_ {
                                            break;
                                        }
                                        phi_35311_ = _e1931_;
                                    }
                                    let _e3536_ = phi_35311_;
                                    if _e197_ {
                                        phi_35650_ = _e3536_;
                                        phi_7873_ = type_15_(0u, 4u);
                                        loop {
                                            let _e4124_ = phi_35650_;
                                            let _e4126_ = phi_7873_;
                                            if (_e4126_.member < _e4126_.member_1_) {
                                                phi_7874_ = type_15_((_e4126_.member + 1u), _e4126_.member_1_);
                                                phi_7897_ = type_15_(1u, _e4126_.member);
                                            } else {
                                                phi_7874_ = _e4126_;
                                                phi_7897_ = type_15_(0u, u32());
                                            }
                                            let _e4139_ = phi_7874_;
                                            let _e4141_ = phi_7897_;
                                            let _e4145_ = (bitcast<i32>(_e4141_.member) != 0i);
                                            if _e4145_ {
                                                let _e4146_ = (_e4141_.member_1_ & 3u);
                                                switch bitcast<i32>(_e4146_) {
                                                    case 0: {
                                                        phi_7915_ = _e370_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_7915_ = _e370_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_7915_ = _e370_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_7915_ = _e370_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4153_ = phi_7915_;
                                                switch bitcast<i32>(_e4146_) {
                                                    case 0: {
                                                        phi_7926_ = _e377_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_7926_ = _e377_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_7926_ = _e377_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_7926_ = _e377_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4160_ = phi_7926_;
                                                switch bitcast<i32>(_e4146_) {
                                                    case 0: {
                                                        phi_7937_ = _e384_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_7937_ = _e384_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_7937_ = _e384_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_7937_ = _e384_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4167_ = phi_7937_;
                                                switch bitcast<i32>(_e4146_) {
                                                    case 0: {
                                                        phi_7948_ = _e391_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_7948_ = _e391_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_7948_ = _e391_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_7948_ = _e391_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4174_ = phi_7948_;
                                                switch bitcast<i32>(_e4146_) {
                                                    case 0: {
                                                        phi_7959_ = _e398_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_7959_ = _e398_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_7959_ = _e398_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_7959_ = _e398_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4181_ = phi_7959_;
                                                switch bitcast<i32>(_e4146_) {
                                                    case 0: {
                                                        phi_7970_ = _e405_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_7970_ = _e405_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_7970_ = _e405_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_7970_ = _e405_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4188_ = phi_7970_;
                                                let _e4189_ = bitcast<u32>(_e4153_);
                                                let _e4190_ = (_e4189_ & 2139095040u);
                                                if (_e4190_ != 2139095040u) {
                                                    if (_e4153_ <= 0f) {
                                                        phi_8073_ = 0f;
                                                    } else {
                                                        if (_e4153_ >= 0.999995f) {
                                                            phi_8072_ = 0.999995f;
                                                        } else {
                                                            if (_e4190_ == 2139095040u) {
                                                                phi_8066_ = true;
                                                            } else {
                                                                phi_8066_ = (_e4190_ == 0u);
                                                            }
                                                            let _e4199_ = phi_8066_;
                                                            phi_8072_ = bitcast<f32>(select(_e4189_, 0u, _e4199_));
                                                        }
                                                        let _e4203_ = phi_8072_;
                                                        phi_8073_ = _e4203_;
                                                    }
                                                    let _e4205_ = phi_8073_;
                                                    phi_8080_ = _e4205_;
                                                } else {
                                                    phi_8080_ = select(0f, 0.999995f, (_e4189_ == 2139095040u));
                                                }
                                                let _e4207_ = phi_8080_;
                                                let _e4208_ = bitcast<u32>(_e4160_);
                                                let _e4209_ = (_e4208_ & 2139095040u);
                                                if (_e4209_ != 2139095040u) {
                                                    if (_e4160_ <= 0f) {
                                                        phi_8112_ = 0f;
                                                    } else {
                                                        if (_e4160_ >= 1f) {
                                                            phi_8111_ = 1f;
                                                        } else {
                                                            if (_e4209_ == 2139095040u) {
                                                                phi_8105_ = true;
                                                            } else {
                                                                phi_8105_ = (_e4209_ == 0u);
                                                            }
                                                            let _e4218_ = phi_8105_;
                                                            phi_8111_ = bitcast<f32>(select(_e4208_, 0u, _e4218_));
                                                        }
                                                        let _e4222_ = phi_8111_;
                                                        phi_8112_ = _e4222_;
                                                    }
                                                    let _e4224_ = phi_8112_;
                                                    phi_8119_ = _e4224_;
                                                } else {
                                                    phi_8119_ = select(0f, 1f, (_e4208_ == 2139095040u));
                                                }
                                                let _e4226_ = phi_8119_;
                                                let _e4227_ = bitcast<u32>(_e4167_);
                                                let _e4228_ = (_e4227_ & 2139095040u);
                                                if (_e4228_ != 2139095040u) {
                                                    if (_e4167_ <= 0f) {
                                                        phi_8151_ = 0f;
                                                    } else {
                                                        if (_e4167_ >= 1f) {
                                                            phi_8150_ = 1f;
                                                        } else {
                                                            if (_e4228_ == 2139095040u) {
                                                                phi_8144_ = true;
                                                            } else {
                                                                phi_8144_ = (_e4228_ == 0u);
                                                            }
                                                            let _e4237_ = phi_8144_;
                                                            phi_8150_ = bitcast<f32>(select(_e4227_, 0u, _e4237_));
                                                        }
                                                        let _e4241_ = phi_8150_;
                                                        phi_8151_ = _e4241_;
                                                    }
                                                    let _e4243_ = phi_8151_;
                                                    phi_8158_ = _e4243_;
                                                } else {
                                                    phi_8158_ = select(0f, 1f, (_e4227_ == 2139095040u));
                                                }
                                                let _e4245_ = phi_8158_;
                                                let _e4246_ = bitcast<u32>(_e4174_);
                                                let _e4247_ = (_e4246_ & 2139095040u);
                                                if (_e4247_ != 2139095040u) {
                                                    if (_e4174_ <= 0f) {
                                                        phi_8190_ = 0f;
                                                    } else {
                                                        if (_e4174_ >= 64f) {
                                                            phi_8189_ = 64f;
                                                        } else {
                                                            if (_e4247_ == 2139095040u) {
                                                                phi_8183_ = true;
                                                            } else {
                                                                phi_8183_ = (_e4247_ == 0u);
                                                            }
                                                            let _e4256_ = phi_8183_;
                                                            phi_8189_ = bitcast<f32>(select(_e4246_, 0u, _e4256_));
                                                        }
                                                        let _e4260_ = phi_8189_;
                                                        phi_8190_ = _e4260_;
                                                    }
                                                    let _e4262_ = phi_8190_;
                                                    phi_8197_ = _e4262_;
                                                } else {
                                                    phi_8197_ = select(0f, 64f, (_e4246_ == 2139095040u));
                                                }
                                                let _e4264_ = phi_8197_;
                                                let _e4265_ = bitcast<u32>(_e4181_);
                                                let _e4266_ = (_e4265_ & 2139095040u);
                                                if (_e4266_ != 2139095040u) {
                                                    if (_e4181_ <= 0f) {
                                                        phi_8229_ = 0f;
                                                    } else {
                                                        if (_e4181_ >= 1f) {
                                                            phi_8228_ = 1f;
                                                        } else {
                                                            if (_e4266_ == 2139095040u) {
                                                                phi_8222_ = true;
                                                            } else {
                                                                phi_8222_ = (_e4266_ == 0u);
                                                            }
                                                            let _e4275_ = phi_8222_;
                                                            phi_8228_ = bitcast<f32>(select(_e4265_, 0u, _e4275_));
                                                        }
                                                        let _e4279_ = phi_8228_;
                                                        phi_8229_ = _e4279_;
                                                    }
                                                    let _e4281_ = phi_8229_;
                                                    phi_8236_ = _e4281_;
                                                } else {
                                                    phi_8236_ = select(0f, 1f, (_e4265_ == 2139095040u));
                                                }
                                                let _e4283_ = phi_8236_;
                                                let _e4284_ = bitcast<u32>(_e4188_);
                                                let _e4285_ = (_e4284_ & 2139095040u);
                                                if (_e4285_ != 2139095040u) {
                                                    if (_e4188_ <= 0f) {
                                                        phi_8268_ = 0f;
                                                    } else {
                                                        if (_e4188_ >= 1f) {
                                                            phi_8267_ = 1f;
                                                        } else {
                                                            if (_e4285_ == 2139095040u) {
                                                                phi_8261_ = true;
                                                            } else {
                                                                phi_8261_ = (_e4285_ == 0u);
                                                            }
                                                            let _e4294_ = phi_8261_;
                                                            phi_8267_ = bitcast<f32>(select(_e4284_, 0u, _e4294_));
                                                        }
                                                        let _e4298_ = phi_8267_;
                                                        phi_8268_ = _e4298_;
                                                    }
                                                    let _e4300_ = phi_8268_;
                                                    phi_8275_ = _e4300_;
                                                } else {
                                                    phi_8275_ = select(0f, 1f, (_e4284_ == 2139095040u));
                                                }
                                                let _e4302_ = phi_8275_;
                                                switch bitcast<i32>(_e4146_) {
                                                    case 0: {
                                                        phi_8346_ = _e412_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_8346_ = _e412_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_8346_ = _e412_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_8346_ = _e412_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4309_ = phi_8346_;
                                                switch bitcast<i32>(_e4146_) {
                                                    case 0: {
                                                        phi_8357_ = _e419_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_8357_ = _e419_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_8357_ = _e419_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_8357_ = _e419_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4316_ = phi_8357_;
                                                if _e1497_ {
                                                    let _e4322_ = ((((_e4141_.member_1_ * 4u) + select(0u, 2u, _e161_)) * 128u) + (_e158_ & 127u));
                                                    if (_e4322_ < 2048u) {
                                                    } else {
                                                        phi_36579_ = true;
                                                        break;
                                                    }
                                                    let _e4325_ = lane_buf[_e4322_];
                                                    phi_8445_ = _e4325_;
                                                } else {
                                                    phi_8445_ = 0f;
                                                }
                                                let _e4327_ = phi_8445_;
                                                if _e1497_ {
                                                    let _e4333_ = ((((_e4141_.member_1_ * 4u) + select(1u, 3u, _e161_)) * 128u) + (_e158_ & 127u));
                                                    if (_e4333_ < 2048u) {
                                                    } else {
                                                        phi_36579_ = true;
                                                        break;
                                                    }
                                                    let _e4336_ = lane_buf[_e4333_];
                                                    phi_8465_ = _e4336_;
                                                } else {
                                                    phi_8465_ = 0f;
                                                }
                                                let _e4338_ = phi_8465_;
                                                let _e4341_ = ((16u + select(0u, 4u, _e161_)) + _e4141_.member_1_);
                                                if (_e4341_ < 256u) {
                                                } else {
                                                    phi_36579_ = true;
                                                    break;
                                                }
                                                let _e4344_ = aux_scratch[_e4341_];
                                                let _e4347_ = (select(512u, 640u, _e161_) + _e158_);
                                                let _e4348_ = (_e4347_ < 1536u);
                                                if _e4348_ {
                                                } else {
                                                    phi_36579_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e4347_] = _e4344_;
                                                workgroupBarrier();
                                                phi_35623_ = _e4124_;
                                                phi_8496_ = type_15_(0u, 6u);
                                                phi_8499_ = _e4344_;
                                                phi_8501_ = false;
                                                loop {
                                                    let _e4351_ = phi_35623_;
                                                    let _e4353_ = phi_8496_;
                                                    let _e4355_ = phi_8499_;
                                                    let _e4357_ = phi_8501_;
                                                    local_37_ = _e4355_;
                                                    local_38_ = _e4355_;
                                                    local_39_ = _e4355_;
                                                    local_40_ = _e4355_;
                                                    if (_e4353_.member < _e4353_.member_1_) {
                                                        phi_8497_ = type_15_((_e4353_.member + 1u), _e4353_.member_1_);
                                                        phi_8524_ = type_15_(1u, _e4353_.member);
                                                    } else {
                                                        phi_8497_ = _e4353_;
                                                        phi_8524_ = type_15_(0u, u32());
                                                    }
                                                    let _e4370_ = phi_8497_;
                                                    let _e4372_ = phi_8524_;
                                                    let _e4375_ = (bitcast<i32>(_e4372_.member) != 0i);
                                                    if _e4375_ {
                                                        if _e4357_ {
                                                            phi_36066_ = _e4351_;
                                                            phi_12409_ = _e4355_;
                                                            phi_12410_ = _e4357_;
                                                        } else {
                                                            let _e4376_ = (_e158_ == 0u);
                                                            if _e4376_ {
                                                                phi_8547_ = _e4344_;
                                                            } else {
                                                                let _e4377_ = (_e4347_ - 1u);
                                                                if (_e4377_ < 1536u) {
                                                                } else {
                                                                    phi_36064_ = true;
                                                                    break;
                                                                }
                                                                let _e4380_ = scan_scratch[_e4377_];
                                                                phi_8547_ = _e4380_;
                                                            }
                                                            let _e4382_ = phi_8547_;
                                                            if _e1497_ {
                                                                let _e4383_ = (_e4316_ * _e4327_);
                                                                let _e4385_ = bitcast<u32>((_e4309_ * _e4338_));
                                                                let _e4386_ = (_e4385_ & 2139095040u);
                                                                if (_e4386_ == 2139095040u) {
                                                                    phi_8577_ = 0u;
                                                                } else {
                                                                    if (_e4386_ == 0u) {
                                                                        phi_8576_ = (_e4385_ & 2147483648u);
                                                                    } else {
                                                                        phi_8576_ = _e4385_;
                                                                    }
                                                                    let _e4391_ = phi_8576_;
                                                                    phi_8577_ = _e4391_;
                                                                }
                                                                let _e4393_ = phi_8577_;
                                                                let _e4399_ = (_e4264_ <= 0f);
                                                                let _e4401_ = (1f - _e4226_);
                                                                let _e4402_ = (_e4401_ + select((_e4264_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e4393_)) & 2147483647u))), 0f, _e4399_));
                                                                if (_e4402_ != _e4402_) {
                                                                    phi_26151_ = true;
                                                                } else {
                                                                    phi_26151_ = (0f >= _e4402_);
                                                                }
                                                                let _e4406_ = phi_26151_;
                                                                let _e4407_ = select(_e4402_, 0f, _e4406_);
                                                                if (_e4407_ != _e4407_) {
                                                                    phi_26211_ = true;
                                                                } else {
                                                                    phi_26211_ = (1f <= _e4407_);
                                                                }
                                                                let _e4411_ = phi_26211_;
                                                                let _e4412_ = select(_e4407_, 1f, _e4411_);
                                                                let _e4413_ = (_e4207_ * _e4382_);
                                                                let _e4414_ = (1f - _e4283_);
                                                                let _e4415_ = (_e4245_ * _e4414_);
                                                                let _e4417_ = (_e4383_ - (_e4415_ * _e4413_));
                                                                if (_e4417_ != _e4417_) {
                                                                    phi_26271_ = true;
                                                                } else {
                                                                    phi_26271_ = (-1f >= _e4417_);
                                                                }
                                                                let _e4421_ = phi_26271_;
                                                                let _e4422_ = select(_e4417_, -1f, _e4421_);
                                                                if (_e4422_ != _e4422_) {
                                                                    phi_26331_ = true;
                                                                } else {
                                                                    phi_26331_ = (1f <= _e4422_);
                                                                }
                                                                let _e4426_ = phi_26331_;
                                                                let _e4427_ = select(_e4422_, 1f, _e4426_);
                                                                let _e4434_ = bitcast<f32>((bitcast<u32>(_e4417_) & 2147483647u));
                                                                if (_e4434_ != _e4434_) {
                                                                    phi_26391_ = true;
                                                                } else {
                                                                    phi_26391_ = (1000000000000000f <= _e4434_);
                                                                }
                                                                let _e4438_ = phi_26391_;
                                                                let _e4439_ = select(_e4434_, 1000000000000000f, _e4438_);
                                                                let _e4440_ = (_e4439_ * _e4439_);
                                                                let _e4441_ = (_e4440_ + 0.0000010000001f);
                                                                if (_e4441_ != _e4441_) {
                                                                    phi_26451_ = true;
                                                                } else {
                                                                    phi_26451_ = (0.000000000001f >= _e4441_);
                                                                }
                                                                let _e4445_ = phi_26451_;
                                                                let _e4446_ = select(_e4441_, 0.000000000001f, _e4445_);
                                                                let _e4447_ = (_e4446_ * 0.5f);
                                                                let _e4452_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e4446_) >> bitcast<u32>(1i))));
                                                                let _e4456_ = (_e4452_ * (1.5f - (_e4447_ * (_e4452_ * _e4452_))));
                                                                let _e4460_ = (_e4456_ * (1.5f - (_e4447_ * (_e4456_ * _e4456_))));
                                                                let _e4465_ = (_e4441_ * (_e4460_ * (1.5f - (_e4447_ * (_e4460_ * _e4460_)))));
                                                                let _e4466_ = (_e4465_ + 0.001f);
                                                                if (_e4466_ != _e4466_) {
                                                                    phi_26511_ = true;
                                                                } else {
                                                                    phi_26511_ = (0.000001f >= _e4466_);
                                                                }
                                                                let _e4470_ = phi_26511_;
                                                                let _e4471_ = select(_e4466_, 0.000001f, _e4470_);
                                                                let _e4474_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4471_)));
                                                                let _e4478_ = (_e4474_ + (_e4474_ * (1f - (_e4471_ * _e4474_))));
                                                                let _e4482_ = (_e4478_ + (_e4478_ * (1f - (_e4471_ * _e4478_))));
                                                                let _e4486_ = (_e4482_ + (_e4482_ * (1f - (_e4471_ * _e4482_))));
                                                                let _e4487_ = (_e4465_ + _e4439_);
                                                                if (_e4487_ != _e4487_) {
                                                                    phi_26571_ = true;
                                                                } else {
                                                                    phi_26571_ = (0.000001f >= _e4487_);
                                                                }
                                                                let _e4491_ = phi_26571_;
                                                                let _e4492_ = select(_e4487_, 0.000001f, _e4491_);
                                                                let _e4495_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4492_)));
                                                                let _e4499_ = (_e4495_ + (_e4495_ * (1f - (_e4492_ * _e4495_))));
                                                                let _e4503_ = (_e4499_ + (_e4499_ * (1f - (_e4492_ * _e4499_))));
                                                                let _e4513_ = (_e4434_ <= 1000000000000000f);
                                                                let _e4523_ = select(select(select(_e4434_, (0.5f * (_e4439_ + (_e4440_ * _e4486_))), _e4513_), select(-0.0005f, -(((0.0005f * (_e4439_ * (_e4503_ + (_e4503_ * (1f - (_e4492_ * _e4503_)))))) * (1f + (_e4439_ * _e4486_)))), _e4513_), (_e4417_ <= 0f)), 0f, (_e4434_ <= 0f));
                                                                let _e4527_ = (_e4283_ <= 0f);
                                                                let _e4534_ = bitcast<u32>((((1f - (_e4283_ * _e4412_)) * _e4413_) + (_e4412_ * select(((_e4414_ * _e4523_) + (_e4283_ * (_e4427_ * (1f - (0.33333334f * (_e4427_ * _e4427_)))))), _e4523_, _e4527_))));
                                                                let _e4535_ = (_e4534_ & 2139095040u);
                                                                if (_e4535_ == 2139095040u) {
                                                                    phi_9319_ = 0u;
                                                                } else {
                                                                    if (_e4535_ == 0u) {
                                                                        phi_9318_ = (_e4534_ & 2147483648u);
                                                                    } else {
                                                                        phi_9318_ = _e4534_;
                                                                    }
                                                                    let _e4540_ = phi_9318_;
                                                                    phi_9319_ = _e4540_;
                                                                }
                                                                let _e4542_ = phi_9319_;
                                                                let _e4544_ = bitcast<u32>(_e4383_);
                                                                let _e4545_ = (_e4544_ & 2139095040u);
                                                                if (_e4545_ == 2139095040u) {
                                                                    phi_9363_ = true;
                                                                } else {
                                                                    phi_9363_ = (_e4545_ == 0u);
                                                                }
                                                                let _e4549_ = phi_9363_;
                                                                let _e4552_ = bitcast<u32>(_e4338_);
                                                                let _e4553_ = (_e4552_ & 2139095040u);
                                                                if (_e4553_ == 2139095040u) {
                                                                    phi_9379_ = true;
                                                                } else {
                                                                    phi_9379_ = (_e4553_ == 0u);
                                                                }
                                                                let _e4557_ = phi_9379_;
                                                                let _e4560_ = bitcast<u32>(_e4382_);
                                                                let _e4561_ = (_e4560_ & 2139095040u);
                                                                if (_e4561_ == 2139095040u) {
                                                                    phi_9395_ = true;
                                                                } else {
                                                                    phi_9395_ = (_e4561_ == 0u);
                                                                }
                                                                let _e4565_ = phi_9395_;
                                                                let _e4569_ = bitcast<u32>((_e4309_ * bitcast<f32>(select(_e4552_, 0u, _e4557_))));
                                                                let _e4570_ = (_e4569_ & 2139095040u);
                                                                if (_e4570_ == 2139095040u) {
                                                                    phi_9412_ = true;
                                                                } else {
                                                                    phi_9412_ = (_e4570_ == 0u);
                                                                }
                                                                let _e4574_ = phi_9412_;
                                                                if _e4399_ {
                                                                    phi_9424_ = 0f;
                                                                } else {
                                                                    phi_9424_ = (_e4264_ * abs(bitcast<f32>(select(_e4569_, 0u, _e4574_))));
                                                                }
                                                                let _e4580_ = phi_9424_;
                                                                let _e4581_ = (_e4401_ + _e4580_);
                                                                if (_e4581_ <= 0f) {
                                                                    phi_9436_ = 0f;
                                                                } else {
                                                                    phi_9436_ = select(_e4581_, 1f, (_e4581_ >= 1f));
                                                                }
                                                                let _e4586_ = phi_9436_;
                                                                let _e4589_ = (bitcast<f32>(select(_e4544_, 0u, _e4549_)) - (_e4415_ * (_e4207_ * bitcast<f32>(select(_e4560_, 0u, _e4565_)))));
                                                                let _e4592_ = bitcast<f32>((bitcast<u32>(_e4589_) & 2147483647u));
                                                                if (_e4592_ != _e4592_) {
                                                                    phi_26751_ = true;
                                                                } else {
                                                                    phi_26751_ = (1000000000000000f <= _e4592_);
                                                                }
                                                                let _e4596_ = phi_26751_;
                                                                let _e4597_ = select(_e4592_, 1000000000000000f, _e4596_);
                                                                let _e4599_ = ((_e4597_ * _e4597_) + 0.0000010000001f);
                                                                if (_e4599_ != _e4599_) {
                                                                    phi_26811_ = true;
                                                                } else {
                                                                    phi_26811_ = (0.000000000001f >= _e4599_);
                                                                }
                                                                let _e4603_ = phi_26811_;
                                                                let _e4604_ = select(_e4599_, 0.000000000001f, _e4603_);
                                                                let _e4605_ = (_e4604_ * 0.5f);
                                                                let _e4610_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e4604_) >> bitcast<u32>(1i))));
                                                                let _e4614_ = (_e4610_ * (1.5f - (_e4605_ * (_e4610_ * _e4610_))));
                                                                let _e4618_ = (_e4614_ * (1.5f - (_e4605_ * (_e4614_ * _e4614_))));
                                                                let _e4622_ = (_e4618_ * (1.5f - (_e4605_ * (_e4618_ * _e4618_))));
                                                                let _e4624_ = ((_e4599_ * _e4622_) + _e4597_);
                                                                if (_e4624_ != _e4624_) {
                                                                    phi_26931_ = true;
                                                                } else {
                                                                    phi_26931_ = (0.000001f >= _e4624_);
                                                                }
                                                                let _e4628_ = phi_26931_;
                                                                let _e4629_ = select(_e4624_, 0.000001f, _e4628_);
                                                                let _e4632_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4629_)));
                                                                let _e4636_ = (_e4632_ + (_e4632_ * (1f - (_e4629_ * _e4632_))));
                                                                let _e4640_ = (_e4636_ + (_e4636_ * (1f - (_e4629_ * _e4636_))));
                                                                let _e4645_ = (_e4592_ <= 1000000000000000f);
                                                                let _e4656_ = select(select(select(1f, (0.5f * (1f + (_e4597_ * _e4622_))), _e4645_), select(0f, ((0.00000050000006f * _e4622_) * (_e4640_ + (_e4640_ * (1f - (_e4629_ * _e4640_))))), _e4645_), (_e4589_ <= 0f)), 0.5f, (_e4592_ <= 0f));
                                                                if (_e4656_ != _e4656_) {
                                                                    phi_26991_ = true;
                                                                } else {
                                                                    phi_26991_ = (0f >= _e4656_);
                                                                }
                                                                let _e4660_ = phi_26991_;
                                                                let _e4661_ = select(_e4656_, 0f, _e4660_);
                                                                if (_e4661_ != _e4661_) {
                                                                    phi_27051_ = true;
                                                                } else {
                                                                    phi_27051_ = (1f <= _e4661_);
                                                                }
                                                                let _e4665_ = phi_27051_;
                                                                let _e4666_ = select(_e4661_, 1f, _e4665_);
                                                                if (abs(_e4589_) < 1f) {
                                                                    phi_9932_ = (1f - (_e4589_ * _e4589_));
                                                                } else {
                                                                    phi_9932_ = 0f;
                                                                }
                                                                let _e4672_ = phi_9932_;
                                                                if _e4527_ {
                                                                    phi_9940_ = _e4666_;
                                                                } else {
                                                                    phi_9940_ = ((_e4414_ * _e4666_) + (_e4283_ * _e4672_));
                                                                }
                                                                let _e4677_ = phi_9940_;
                                                                let _e4684_ = (_e4207_ * ((1f - (_e4283_ * _e4586_)) - (((_e4586_ * _e4245_) * _e4414_) * _e4677_)));
                                                                let _e4685_ = bitcast<u32>(_e4684_);
                                                                let _e4686_ = (_e4685_ & 2139095040u);
                                                                let _e4687_ = (_e4686_ == 2139095040u);
                                                                if _e4687_ {
                                                                    if ((_e4685_ & 2147483647u) > 2139095040u) {
                                                                        phi_9997_ = f32();
                                                                        phi_9998_ = true;
                                                                    } else {
                                                                        let _e4702_ = ((_e4685_ & 2147483648u) == 0u);
                                                                        if _e4702_ {
                                                                            phi_9995_ = 0.999995f;
                                                                        } else {
                                                                            phi_9995_ = f32();
                                                                        }
                                                                        let _e4704_ = phi_9995_;
                                                                        phi_9997_ = _e4704_;
                                                                        phi_9998_ = select(true, false, _e4702_);
                                                                    }
                                                                    let _e4707_ = phi_9997_;
                                                                    let _e4709_ = phi_9998_;
                                                                    if _e4709_ {
                                                                        phi_10018_ = -0.999995f;
                                                                    } else {
                                                                        phi_10018_ = _e4707_;
                                                                    }
                                                                    let _e4711_ = phi_10018_;
                                                                    phi_10077_ = _e4711_;
                                                                } else {
                                                                    if (_e4684_ < -0.999995f) {
                                                                        phi_10076_ = -0.999995f;
                                                                    } else {
                                                                        if (_e4684_ > 0.999995f) {
                                                                            phi_10075_ = 0.999995f;
                                                                        } else {
                                                                            if _e4687_ {
                                                                                phi_10069_ = true;
                                                                            } else {
                                                                                phi_10069_ = (_e4686_ == 0u);
                                                                            }
                                                                            let _e4692_ = phi_10069_;
                                                                            phi_10075_ = bitcast<f32>(select(_e4685_, 0u, _e4692_));
                                                                        }
                                                                        let _e4696_ = phi_10075_;
                                                                        phi_10076_ = _e4696_;
                                                                    }
                                                                    let _e4698_ = phi_10076_;
                                                                    phi_10077_ = _e4698_;
                                                                }
                                                                let _e4713_ = phi_10077_;
                                                                phi_10080_ = type_28_(bitcast<f32>(_e4542_), _e4713_);
                                                            } else {
                                                                phi_10080_ = type_28_(0f, 1f);
                                                            }
                                                            let _e4716_ = phi_10080_;
                                                            if _e1497_ {
                                                                if ((bitcast<u32>(_e4716_.member) & 2139095040u) != 2139095040u) {
                                                                    let _e4724_ = ((bitcast<u32>(_e4355_) & 2139095040u) != 2139095040u);
                                                                    if _e4724_ {
                                                                        let _e4725_ = abs(_e4716_.member);
                                                                        let _e4726_ = abs(_e4355_);
                                                                        let _e4728_ = select(_e4726_, _e4725_, (_e4725_ > _e4726_));
                                                                        let _e4730_ = select(1f, _e4728_, (_e4728_ > 1f));
                                                                        if ((bitcast<u32>(_e4730_) & 2139095040u) != 2139095040u) {
                                                                            let _e4736_ = (_e4730_ <= 0f);
                                                                            if _e4736_ {
                                                                                phi_10204_ = f32();
                                                                            } else {
                                                                                if (_e4730_ < 0.00001f) {
                                                                                    let _e4768_ = (select(0.000000000000000000000001f, _e4730_, (_e4730_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                    if (_e4768_ != _e4768_) {
                                                                                        phi_27171_ = true;
                                                                                    } else {
                                                                                        phi_27171_ = (0.000001f >= _e4768_);
                                                                                    }
                                                                                    let _e4772_ = phi_27171_;
                                                                                    let _e4773_ = select(_e4768_, 0.000001f, _e4772_);
                                                                                    let _e4776_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4773_)));
                                                                                    let _e4780_ = (_e4776_ + (_e4776_ * (1f - (_e4773_ * _e4776_))));
                                                                                    let _e4784_ = (_e4780_ + (_e4780_ * (1f - (_e4773_ * _e4780_))));
                                                                                    let _e4790_ = bitcast<u32>(((_e4784_ + (_e4784_ * (1f - (_e4773_ * _e4784_)))) * 18446744000000000000f));
                                                                                    let _e4791_ = (_e4790_ & 2139095040u);
                                                                                    if (_e4791_ == 2139095040u) {
                                                                                        phi_10162_ = true;
                                                                                    } else {
                                                                                        phi_10162_ = (_e4791_ == 0u);
                                                                                    }
                                                                                    let _e4795_ = phi_10162_;
                                                                                    phi_10203_ = bitcast<f32>(select(_e4790_, 0u, _e4795_));
                                                                                } else {
                                                                                    if (_e4730_ != _e4730_) {
                                                                                        phi_27111_ = true;
                                                                                    } else {
                                                                                        phi_27111_ = (0.000001f >= _e4730_);
                                                                                    }
                                                                                    let _e4741_ = phi_27111_;
                                                                                    let _e4742_ = select(_e4730_, 0.000001f, _e4741_);
                                                                                    let _e4745_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4742_)));
                                                                                    let _e4749_ = (_e4745_ + (_e4745_ * (1f - (_e4742_ * _e4745_))));
                                                                                    let _e4753_ = (_e4749_ + (_e4749_ * (1f - (_e4742_ * _e4749_))));
                                                                                    let _e4758_ = bitcast<u32>((_e4753_ + (_e4753_ * (1f - (_e4742_ * _e4753_)))));
                                                                                    let _e4759_ = (_e4758_ & 2139095040u);
                                                                                    if (_e4759_ == 2139095040u) {
                                                                                        phi_10197_ = true;
                                                                                    } else {
                                                                                        phi_10197_ = (_e4759_ == 0u);
                                                                                    }
                                                                                    let _e4763_ = phi_10197_;
                                                                                    phi_10203_ = bitcast<f32>(select(_e4758_, 0u, _e4763_));
                                                                                }
                                                                                let _e4799_ = phi_10203_;
                                                                                phi_10204_ = _e4799_;
                                                                            }
                                                                            let _e4801_ = phi_10204_;
                                                                            phi_10205_ = _e4801_;
                                                                            phi_10206_ = _e4736_;
                                                                        } else {
                                                                            phi_10205_ = f32();
                                                                            phi_10206_ = true;
                                                                        }
                                                                        let _e4803_ = phi_10205_;
                                                                        let _e4805_ = phi_10206_;
                                                                        phi_10212_ = (abs((_e4716_.member - _e4355_)) * select(_e4803_, 0f, _e4805_));
                                                                    } else {
                                                                        phi_10212_ = f32();
                                                                    }
                                                                    let _e4809_ = phi_10212_;
                                                                    phi_10214_ = _e4809_;
                                                                    phi_10215_ = select(true, false, _e4724_);
                                                                } else {
                                                                    phi_10214_ = f32();
                                                                    phi_10215_ = true;
                                                                }
                                                                let _e4812_ = phi_10214_;
                                                                let _e4814_ = phi_10215_;
                                                                phi_10220_ = select(_e4812_, 1000000000f, _e4814_);
                                                            } else {
                                                                phi_10220_ = 0f;
                                                            }
                                                            let _e4817_ = phi_10220_;
                                                            let _e4818_ = (_e157_ < 128u);
                                                            if _e4818_ {
                                                                let _e4819_ = (32u + _e158_);
                                                                if (_e4819_ < 256u) {
                                                                } else {
                                                                    phi_36064_ = true;
                                                                    break;
                                                                }
                                                                aux_scratch[_e4819_] = _e4817_;
                                                            }
                                                            workgroupBarrier();
                                                            let _e4822_ = (_e157_ >= 128u);
                                                            if _e4822_ {
                                                                let _e4823_ = (32u + _e158_);
                                                                let _e4824_ = (_e4823_ < 256u);
                                                                if _e4824_ {
                                                                } else {
                                                                    phi_36064_ = true;
                                                                    break;
                                                                }
                                                                let _e4826_ = aux_scratch[_e4823_];
                                                                if _e4824_ {
                                                                } else {
                                                                    phi_36064_ = true;
                                                                    break;
                                                                }
                                                                aux_scratch[_e4823_] = select(_e4826_, _e4817_, (_e4817_ > _e4826_));
                                                            }
                                                            workgroupBarrier();
                                                            phi_10258_ = 64u;
                                                            loop {
                                                                let _e4830_ = phi_10258_;
                                                                let _e4831_ = (_e4830_ > 0u);
                                                                if _e4831_ {
                                                                    if (_e158_ < _e4830_) {
                                                                        if _e4818_ {
                                                                            let _e4833_ = (32u + _e158_);
                                                                            let _e4834_ = (_e4833_ < 256u);
                                                                            if _e4834_ {
                                                                            } else {
                                                                                phi_35614_ = true;
                                                                                break;
                                                                            }
                                                                            let _e4836_ = aux_scratch[_e4833_];
                                                                            let _e4837_ = (_e4833_ + _e4830_);
                                                                            if (_e4837_ < 256u) {
                                                                            } else {
                                                                                phi_35614_ = true;
                                                                                break;
                                                                            }
                                                                            let _e4840_ = aux_scratch[_e4837_];
                                                                            if _e4834_ {
                                                                            } else {
                                                                                phi_35614_ = true;
                                                                                break;
                                                                            }
                                                                            aux_scratch[_e4833_] = select(_e4836_, _e4840_, (_e4840_ > _e4836_));
                                                                        }
                                                                    }
                                                                    workgroupBarrier();
                                                                    phi_10259_ = (_e4830_ >> bitcast<u32>(1i));
                                                                } else {
                                                                    phi_10259_ = u32();
                                                                }
                                                                let _e4846_ = phi_10259_;
                                                                phi_10258_ = _e4846_;
                                                                phi_35614_ = _e4351_;
                                                                if !(_e4831_) {
                                                                    break;
                                                                }
                                                            }
                                                            let _e4849_ = phi_35614_;
                                                            phi_36064_ = _e4849_;
                                                            if _e4849_ {
                                                                break;
                                                            }
                                                            let _e4851_ = aux_scratch[32];
                                                            workgroupBarrier();
                                                            if (_e4851_ <= 0.0000005f) {
                                                                phi_36067_ = _e4849_;
                                                                phi_12407_ = _e4355_;
                                                                phi_12408_ = true;
                                                            } else {
                                                                if _e1497_ {
                                                                    phi_10318_ = type_28_(_e4716_.member_1_, (_e4716_.member - (_e4716_.member_1_ * _e4382_)));
                                                                } else {
                                                                    phi_10318_ = type_28_(1f, 0f);
                                                                }
                                                                let _e4857_ = phi_10318_;
                                                                let _e4860_ = select(0u, 256u, _e161_);
                                                                let _e4861_ = (_e158_ & 127u);
                                                                let _e4862_ = (_e4860_ + _e4861_);
                                                                let _e4863_ = (_e4862_ < 1536u);
                                                                if _e4863_ {
                                                                } else {
                                                                    phi_36064_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e4862_] = _e4857_.member;
                                                                let _e4865_ = (_e4860_ + 128u);
                                                                let _e4866_ = (_e4865_ + _e4861_);
                                                                let _e4867_ = (_e4866_ < 1536u);
                                                                if _e4867_ {
                                                                } else {
                                                                    phi_36064_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e4866_] = _e4857_.member_1_;
                                                                workgroupBarrier();
                                                                phi_10346_ = 1u;
                                                                phi_10349_ = _e4857_;
                                                                loop {
                                                                    let _e4870_ = phi_10346_;
                                                                    let _e4872_ = phi_10349_;
                                                                    local_35_ = _e4872_;
                                                                    local_36_ = _e4872_;
                                                                    let _e4873_ = (_e4870_ < 128u);
                                                                    if _e4873_ {
                                                                        if (_e158_ >= _e4870_) {
                                                                            let _e4876_ = ((_e158_ - _e4870_) & 127u);
                                                                            let _e4877_ = (_e4860_ + _e4876_);
                                                                            if (_e4877_ < 1536u) {
                                                                            } else {
                                                                                phi_35904_ = true;
                                                                                break;
                                                                            }
                                                                            let _e4880_ = scan_scratch[_e4877_];
                                                                            let _e4881_ = (_e4865_ + _e4876_);
                                                                            if (_e4881_ < 1536u) {
                                                                            } else {
                                                                                phi_35904_ = true;
                                                                                break;
                                                                            }
                                                                            let _e4884_ = scan_scratch[_e4881_];
                                                                            phi_10389_ = type_28_((_e4872_.member * _e4880_), ((_e4872_.member * _e4884_) + _e4872_.member_1_));
                                                                        } else {
                                                                            phi_10389_ = _e4872_;
                                                                        }
                                                                        let _e4892_ = phi_10389_;
                                                                        workgroupBarrier();
                                                                        if _e4863_ {
                                                                        } else {
                                                                            phi_35904_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e4862_] = _e4892_.member;
                                                                        if _e4867_ {
                                                                        } else {
                                                                            phi_35904_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e4866_] = _e4892_.member_1_;
                                                                        workgroupBarrier();
                                                                        phi_10347_ = (_e4870_ << bitcast<u32>(1i));
                                                                        phi_10350_ = _e4892_;
                                                                    } else {
                                                                        phi_10347_ = u32();
                                                                        phi_10350_ = type_28_();
                                                                    }
                                                                    let _e4898_ = phi_10347_;
                                                                    let _e4900_ = phi_10350_;
                                                                    phi_10346_ = _e4898_;
                                                                    phi_10349_ = _e4900_;
                                                                    phi_35904_ = _e4849_;
                                                                    if !(_e4873_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e4903_ = phi_35904_;
                                                                phi_36064_ = _e4903_;
                                                                if _e4903_ {
                                                                    break;
                                                                }
                                                                let _e4905_ = local_35_;
                                                                let _e4908_ = local_36_;
                                                                if _e1496_ {
                                                                    if (_e1495_ <= 128u) {
                                                                        phi_10425_ = (_e1495_ - 1u);
                                                                    } else {
                                                                        phi_10425_ = 127u;
                                                                    }
                                                                    let _e4915_ = phi_10425_;
                                                                    let _e4916_ = (_e4915_ & 127u);
                                                                    if ((_e4860_ + _e4916_) < 1536u) {
                                                                    } else {
                                                                        phi_36064_ = true;
                                                                        break;
                                                                    }
                                                                    if ((_e4865_ + _e4916_) < 1536u) {
                                                                    } else {
                                                                        phi_36064_ = true;
                                                                        break;
                                                                    }
                                                                }
                                                                workgroupBarrier();
                                                                phi_35919_ = _e4903_;
                                                                phi_10447_ = type_33_(0i, 3i);
                                                                phi_10450_ = _e4355_;
                                                                phi_10452_ = 1f;
                                                                phi_10454_ = false;
                                                                phi_10456_ = _e4357_;
                                                                loop {
                                                                    let _e4923_ = phi_35919_;
                                                                    let _e4925_ = phi_10447_;
                                                                    let _e4927_ = phi_10450_;
                                                                    let _e4929_ = phi_10452_;
                                                                    let _e4931_ = phi_10454_;
                                                                    let _e4933_ = phi_10456_;
                                                                    local_45_ = _e4927_;
                                                                    local_46_ = _e4933_;
                                                                    if (_e4925_.member < _e4925_.member_1_) {
                                                                        let _e4940_ = (_e4925_.member + 1i);
                                                                        if select(false, true, ((false == (_e4940_ > _e4925_.member)) != false)) {
                                                                            phi_10483_ = type_34_(0u, i32());
                                                                        } else {
                                                                            phi_10483_ = type_34_(1u, _e4940_);
                                                                        }
                                                                        let _e4950_ = phi_10483_;
                                                                        if (bitcast<i32>(_e4950_.member) != 0i) {
                                                                        } else {
                                                                            phi_36058_ = true;
                                                                            break;
                                                                        }
                                                                        phi_10448_ = type_33_(_e4950_.member_1_, _e4925_.member_1_);
                                                                        phi_10497_ = type_34_(1u, _e4925_.member);
                                                                    } else {
                                                                        phi_10448_ = _e4925_;
                                                                        phi_10497_ = type_34_(0u, i32());
                                                                    }
                                                                    let _e4960_ = phi_10448_;
                                                                    let _e4962_ = phi_10497_;
                                                                    let _e4966_ = (bitcast<i32>(_e4962_.member) != 0i);
                                                                    if _e4966_ {
                                                                        if _e4931_ {
                                                                            phi_36060_ = _e4923_;
                                                                            phi_12403_ = _e4927_;
                                                                            phi_12404_ = _e4929_;
                                                                            phi_12405_ = _e4931_;
                                                                            phi_12406_ = _e4933_;
                                                                        } else {
                                                                            if _e1497_ {
                                                                                let _e4968_ = (_e4927_ + (_e4929_ * (((_e4905_.member * _e4344_) + _e4908_.member_1_) - _e4355_)));
                                                                                let _e4969_ = bitcast<u32>(_e4968_);
                                                                                let _e4970_ = (_e4969_ & 2139095040u);
                                                                                let _e4971_ = (_e4970_ == 2139095040u);
                                                                                if _e4971_ {
                                                                                    if ((_e4969_ & 2147483647u) > 2139095040u) {
                                                                                        phi_10557_ = f32();
                                                                                        phi_10558_ = true;
                                                                                    } else {
                                                                                        let _e4986_ = ((_e4969_ & 2147483648u) == 0u);
                                                                                        if _e4986_ {
                                                                                            phi_10555_ = 1000000f;
                                                                                        } else {
                                                                                            phi_10555_ = f32();
                                                                                        }
                                                                                        let _e4988_ = phi_10555_;
                                                                                        phi_10557_ = _e4988_;
                                                                                        phi_10558_ = select(true, false, _e4986_);
                                                                                    }
                                                                                    let _e4991_ = phi_10557_;
                                                                                    let _e4993_ = phi_10558_;
                                                                                    if _e4993_ {
                                                                                        phi_10578_ = -1000000f;
                                                                                    } else {
                                                                                        phi_10578_ = _e4991_;
                                                                                    }
                                                                                    let _e4995_ = phi_10578_;
                                                                                    phi_10637_ = _e4995_;
                                                                                } else {
                                                                                    if (_e4968_ < -1000000f) {
                                                                                        phi_10636_ = -1000000f;
                                                                                    } else {
                                                                                        if (_e4968_ > 1000000f) {
                                                                                            phi_10635_ = 1000000f;
                                                                                        } else {
                                                                                            if _e4971_ {
                                                                                                phi_10629_ = true;
                                                                                            } else {
                                                                                                phi_10629_ = (_e4970_ == 0u);
                                                                                            }
                                                                                            let _e4976_ = phi_10629_;
                                                                                            phi_10635_ = bitcast<f32>(select(_e4969_, 0u, _e4976_));
                                                                                        }
                                                                                        let _e4980_ = phi_10635_;
                                                                                        phi_10636_ = _e4980_;
                                                                                    }
                                                                                    let _e4982_ = phi_10636_;
                                                                                    phi_10637_ = _e4982_;
                                                                                }
                                                                                let _e4997_ = phi_10637_;
                                                                                phi_10638_ = _e4997_;
                                                                            } else {
                                                                                phi_10638_ = _e4344_;
                                                                            }
                                                                            let _e4999_ = phi_10638_;
                                                                            let _e5000_ = (select(768u, 896u, _e161_) + _e158_);
                                                                            if (_e5000_ < 1536u) {
                                                                            } else {
                                                                                phi_36058_ = true;
                                                                                break;
                                                                            }
                                                                            scan_scratch[_e5000_] = _e4999_;
                                                                            workgroupBarrier();
                                                                            if _e4376_ {
                                                                                phi_10659_ = _e4344_;
                                                                            } else {
                                                                                let _e5003_ = (_e5000_ - 1u);
                                                                                if (_e5003_ < 1536u) {
                                                                                } else {
                                                                                    phi_36058_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e5006_ = scan_scratch[_e5003_];
                                                                                phi_10659_ = _e5006_;
                                                                            }
                                                                            let _e5008_ = phi_10659_;
                                                                            if _e1497_ {
                                                                                let _e5011_ = bitcast<u32>((_e4309_ * _e4338_));
                                                                                let _e5012_ = (_e5011_ & 2139095040u);
                                                                                if (_e5012_ == 2139095040u) {
                                                                                    phi_10689_ = 0u;
                                                                                } else {
                                                                                    if (_e5012_ == 0u) {
                                                                                        phi_10688_ = (_e5011_ & 2147483648u);
                                                                                    } else {
                                                                                        phi_10688_ = _e5011_;
                                                                                    }
                                                                                    let _e5017_ = phi_10688_;
                                                                                    phi_10689_ = _e5017_;
                                                                                }
                                                                                let _e5019_ = phi_10689_;
                                                                                let _e5028_ = ((1f - _e4226_) + select((_e4264_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e5019_)) & 2147483647u))), 0f, (_e4264_ <= 0f)));
                                                                                if (_e5028_ != _e5028_) {
                                                                                    phi_27231_ = true;
                                                                                } else {
                                                                                    phi_27231_ = (0f >= _e5028_);
                                                                                }
                                                                                let _e5032_ = phi_27231_;
                                                                                let _e5033_ = select(_e5028_, 0f, _e5032_);
                                                                                if (_e5033_ != _e5033_) {
                                                                                    phi_27291_ = true;
                                                                                } else {
                                                                                    phi_27291_ = (1f <= _e5033_);
                                                                                }
                                                                                let _e5037_ = phi_27291_;
                                                                                let _e5038_ = select(_e5033_, 1f, _e5037_);
                                                                                let _e5039_ = (_e4207_ * _e5008_);
                                                                                let _e5040_ = (1f - _e4283_);
                                                                                let _e5043_ = ((_e4316_ * _e4327_) - ((_e4245_ * _e5040_) * _e5039_));
                                                                                if (_e5043_ != _e5043_) {
                                                                                    phi_27351_ = true;
                                                                                } else {
                                                                                    phi_27351_ = (-1f >= _e5043_);
                                                                                }
                                                                                let _e5047_ = phi_27351_;
                                                                                let _e5048_ = select(_e5043_, -1f, _e5047_);
                                                                                if (_e5048_ != _e5048_) {
                                                                                    phi_27411_ = true;
                                                                                } else {
                                                                                    phi_27411_ = (1f <= _e5048_);
                                                                                }
                                                                                let _e5052_ = phi_27411_;
                                                                                let _e5053_ = select(_e5048_, 1f, _e5052_);
                                                                                let _e5060_ = bitcast<f32>((bitcast<u32>(_e5043_) & 2147483647u));
                                                                                if (_e5060_ != _e5060_) {
                                                                                    phi_27471_ = true;
                                                                                } else {
                                                                                    phi_27471_ = (1000000000000000f <= _e5060_);
                                                                                }
                                                                                let _e5064_ = phi_27471_;
                                                                                let _e5065_ = select(_e5060_, 1000000000000000f, _e5064_);
                                                                                let _e5066_ = (_e5065_ * _e5065_);
                                                                                let _e5067_ = (_e5066_ + 0.0000010000001f);
                                                                                if (_e5067_ != _e5067_) {
                                                                                    phi_27531_ = true;
                                                                                } else {
                                                                                    phi_27531_ = (0.000000000001f >= _e5067_);
                                                                                }
                                                                                let _e5071_ = phi_27531_;
                                                                                let _e5072_ = select(_e5067_, 0.000000000001f, _e5071_);
                                                                                let _e5073_ = (_e5072_ * 0.5f);
                                                                                let _e5078_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e5072_) >> bitcast<u32>(1i))));
                                                                                let _e5082_ = (_e5078_ * (1.5f - (_e5073_ * (_e5078_ * _e5078_))));
                                                                                let _e5086_ = (_e5082_ * (1.5f - (_e5073_ * (_e5082_ * _e5082_))));
                                                                                let _e5091_ = (_e5067_ * (_e5086_ * (1.5f - (_e5073_ * (_e5086_ * _e5086_)))));
                                                                                let _e5092_ = (_e5091_ + 0.001f);
                                                                                if (_e5092_ != _e5092_) {
                                                                                    phi_27591_ = true;
                                                                                } else {
                                                                                    phi_27591_ = (0.000001f >= _e5092_);
                                                                                }
                                                                                let _e5096_ = phi_27591_;
                                                                                let _e5097_ = select(_e5092_, 0.000001f, _e5096_);
                                                                                let _e5100_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5097_)));
                                                                                let _e5104_ = (_e5100_ + (_e5100_ * (1f - (_e5097_ * _e5100_))));
                                                                                let _e5108_ = (_e5104_ + (_e5104_ * (1f - (_e5097_ * _e5104_))));
                                                                                let _e5112_ = (_e5108_ + (_e5108_ * (1f - (_e5097_ * _e5108_))));
                                                                                let _e5113_ = (_e5091_ + _e5065_);
                                                                                if (_e5113_ != _e5113_) {
                                                                                    phi_27651_ = true;
                                                                                } else {
                                                                                    phi_27651_ = (0.000001f >= _e5113_);
                                                                                }
                                                                                let _e5117_ = phi_27651_;
                                                                                let _e5118_ = select(_e5113_, 0.000001f, _e5117_);
                                                                                let _e5121_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5118_)));
                                                                                let _e5125_ = (_e5121_ + (_e5121_ * (1f - (_e5118_ * _e5121_))));
                                                                                let _e5129_ = (_e5125_ + (_e5125_ * (1f - (_e5118_ * _e5125_))));
                                                                                let _e5139_ = (_e5060_ <= 1000000000000000f);
                                                                                let _e5149_ = select(select(select(_e5060_, (0.5f * (_e5065_ + (_e5066_ * _e5112_))), _e5139_), select(-0.0005f, -(((0.0005f * (_e5065_ * (_e5129_ + (_e5129_ * (1f - (_e5118_ * _e5129_)))))) * (1f + (_e5065_ * _e5112_)))), _e5139_), (_e5043_ <= 0f)), 0f, (_e5060_ <= 0f));
                                                                                let _e5160_ = bitcast<u32>((((1f - (_e4283_ * _e5038_)) * _e5039_) + (_e5038_ * select(((_e5040_ * _e5149_) + (_e4283_ * (_e5053_ * (1f - (0.33333334f * (_e5053_ * _e5053_)))))), _e5149_, (_e4283_ <= 0f)))));
                                                                                let _e5161_ = (_e5160_ & 2139095040u);
                                                                                if (_e5161_ == 2139095040u) {
                                                                                    phi_11431_ = 0u;
                                                                                } else {
                                                                                    if (_e5161_ == 0u) {
                                                                                        phi_11430_ = (_e5160_ & 2147483648u);
                                                                                    } else {
                                                                                        phi_11430_ = _e5160_;
                                                                                    }
                                                                                    let _e5166_ = phi_11430_;
                                                                                    phi_11431_ = _e5166_;
                                                                                }
                                                                                let _e5168_ = phi_11431_;
                                                                                let _e5169_ = bitcast<f32>(_e5168_);
                                                                                if ((bitcast<u32>(_e5169_) & 2139095040u) != 2139095040u) {
                                                                                    let _e5175_ = ((bitcast<u32>(_e4999_) & 2139095040u) != 2139095040u);
                                                                                    if _e5175_ {
                                                                                        let _e5176_ = abs(_e5169_);
                                                                                        let _e5177_ = abs(_e4999_);
                                                                                        let _e5179_ = select(_e5177_, _e5176_, (_e5176_ > _e5177_));
                                                                                        let _e5181_ = select(1f, _e5179_, (_e5179_ > 1f));
                                                                                        if ((bitcast<u32>(_e5181_) & 2139095040u) != 2139095040u) {
                                                                                            let _e5187_ = (_e5181_ <= 0f);
                                                                                            if _e5187_ {
                                                                                                phi_12272_ = f32();
                                                                                            } else {
                                                                                                if (_e5181_ < 0.00001f) {
                                                                                                    let _e5219_ = (select(0.000000000000000000000001f, _e5181_, (_e5181_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                                    if (_e5219_ != _e5219_) {
                                                                                                        phi_28251_ = true;
                                                                                                    } else {
                                                                                                        phi_28251_ = (0.000001f >= _e5219_);
                                                                                                    }
                                                                                                    let _e5223_ = phi_28251_;
                                                                                                    let _e5224_ = select(_e5219_, 0.000001f, _e5223_);
                                                                                                    let _e5227_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5224_)));
                                                                                                    let _e5231_ = (_e5227_ + (_e5227_ * (1f - (_e5224_ * _e5227_))));
                                                                                                    let _e5235_ = (_e5231_ + (_e5231_ * (1f - (_e5224_ * _e5231_))));
                                                                                                    let _e5241_ = bitcast<u32>(((_e5235_ + (_e5235_ * (1f - (_e5224_ * _e5235_)))) * 18446744000000000000f));
                                                                                                    let _e5242_ = (_e5241_ & 2139095040u);
                                                                                                    if (_e5242_ == 2139095040u) {
                                                                                                        phi_12230_ = true;
                                                                                                    } else {
                                                                                                        phi_12230_ = (_e5242_ == 0u);
                                                                                                    }
                                                                                                    let _e5246_ = phi_12230_;
                                                                                                    phi_12271_ = bitcast<f32>(select(_e5241_, 0u, _e5246_));
                                                                                                } else {
                                                                                                    if (_e5181_ != _e5181_) {
                                                                                                        phi_28191_ = true;
                                                                                                    } else {
                                                                                                        phi_28191_ = (0.000001f >= _e5181_);
                                                                                                    }
                                                                                                    let _e5192_ = phi_28191_;
                                                                                                    let _e5193_ = select(_e5181_, 0.000001f, _e5192_);
                                                                                                    let _e5196_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5193_)));
                                                                                                    let _e5200_ = (_e5196_ + (_e5196_ * (1f - (_e5193_ * _e5196_))));
                                                                                                    let _e5204_ = (_e5200_ + (_e5200_ * (1f - (_e5193_ * _e5200_))));
                                                                                                    let _e5209_ = bitcast<u32>((_e5204_ + (_e5204_ * (1f - (_e5193_ * _e5204_)))));
                                                                                                    let _e5210_ = (_e5209_ & 2139095040u);
                                                                                                    if (_e5210_ == 2139095040u) {
                                                                                                        phi_12265_ = true;
                                                                                                    } else {
                                                                                                        phi_12265_ = (_e5210_ == 0u);
                                                                                                    }
                                                                                                    let _e5214_ = phi_12265_;
                                                                                                    phi_12271_ = bitcast<f32>(select(_e5209_, 0u, _e5214_));
                                                                                                }
                                                                                                let _e5250_ = phi_12271_;
                                                                                                phi_12272_ = _e5250_;
                                                                                            }
                                                                                            let _e5252_ = phi_12272_;
                                                                                            phi_12273_ = _e5252_;
                                                                                            phi_12274_ = _e5187_;
                                                                                        } else {
                                                                                            phi_12273_ = f32();
                                                                                            phi_12274_ = true;
                                                                                        }
                                                                                        let _e5254_ = phi_12273_;
                                                                                        let _e5256_ = phi_12274_;
                                                                                        phi_12280_ = (abs((_e5169_ - _e4999_)) * select(_e5254_, 0f, _e5256_));
                                                                                    } else {
                                                                                        phi_12280_ = f32();
                                                                                    }
                                                                                    let _e5260_ = phi_12280_;
                                                                                    phi_12282_ = _e5260_;
                                                                                    phi_12283_ = select(true, false, _e5175_);
                                                                                } else {
                                                                                    phi_12282_ = f32();
                                                                                    phi_12283_ = true;
                                                                                }
                                                                                let _e5263_ = phi_12282_;
                                                                                let _e5265_ = phi_12283_;
                                                                                phi_12288_ = select(_e5263_, 1000000000f, _e5265_);
                                                                            } else {
                                                                                phi_12288_ = 0f;
                                                                            }
                                                                            let _e5268_ = phi_12288_;
                                                                            if _e4818_ {
                                                                                let _e5269_ = (32u + _e158_);
                                                                                if (_e5269_ < 256u) {
                                                                                } else {
                                                                                    phi_36058_ = true;
                                                                                    break;
                                                                                }
                                                                                aux_scratch[_e5269_] = _e5268_;
                                                                            }
                                                                            workgroupBarrier();
                                                                            if _e4822_ {
                                                                                let _e5272_ = (32u + _e158_);
                                                                                let _e5273_ = (_e5272_ < 256u);
                                                                                if _e5273_ {
                                                                                } else {
                                                                                    phi_36058_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e5275_ = aux_scratch[_e5272_];
                                                                                if _e5273_ {
                                                                                } else {
                                                                                    phi_36058_ = true;
                                                                                    break;
                                                                                }
                                                                                aux_scratch[_e5272_] = select(_e5275_, _e5268_, (_e5268_ > _e5275_));
                                                                            }
                                                                            workgroupBarrier();
                                                                            phi_12326_ = 64u;
                                                                            loop {
                                                                                let _e5279_ = phi_12326_;
                                                                                let _e5280_ = (_e5279_ > 0u);
                                                                                if _e5280_ {
                                                                                    if (_e158_ < _e5279_) {
                                                                                        if _e4818_ {
                                                                                            let _e5282_ = (32u + _e158_);
                                                                                            let _e5283_ = (_e5282_ < 256u);
                                                                                            if _e5283_ {
                                                                                            } else {
                                                                                                phi_35910_ = true;
                                                                                                break;
                                                                                            }
                                                                                            let _e5285_ = aux_scratch[_e5282_];
                                                                                            let _e5286_ = (_e5282_ + _e5279_);
                                                                                            if (_e5286_ < 256u) {
                                                                                            } else {
                                                                                                phi_35910_ = true;
                                                                                                break;
                                                                                            }
                                                                                            let _e5289_ = aux_scratch[_e5286_];
                                                                                            if _e5283_ {
                                                                                            } else {
                                                                                                phi_35910_ = true;
                                                                                                break;
                                                                                            }
                                                                                            aux_scratch[_e5282_] = select(_e5285_, _e5289_, (_e5289_ > _e5285_));
                                                                                        }
                                                                                    }
                                                                                    workgroupBarrier();
                                                                                    phi_12327_ = (_e5279_ >> bitcast<u32>(1i));
                                                                                } else {
                                                                                    phi_12327_ = u32();
                                                                                }
                                                                                let _e5295_ = phi_12327_;
                                                                                phi_12326_ = _e5295_;
                                                                                phi_35910_ = _e4923_;
                                                                                if !(_e5280_) {
                                                                                    break;
                                                                                }
                                                                            }
                                                                            let _e5298_ = phi_35910_;
                                                                            phi_36058_ = _e5298_;
                                                                            if _e5298_ {
                                                                                break;
                                                                            }
                                                                            let _e5299_ = aux_scratch[32];
                                                                            workgroupBarrier();
                                                                            if (_e5299_ <= _e4851_) {
                                                                                phi_12383_ = true;
                                                                            } else {
                                                                                phi_12383_ = (_e4962_.member_1_ == 2i);
                                                                            }
                                                                            let _e5303_ = phi_12383_;
                                                                            if _e5303_ {
                                                                                if _e4348_ {
                                                                                } else {
                                                                                    phi_36058_ = true;
                                                                                    break;
                                                                                }
                                                                                scan_scratch[_e4347_] = _e4999_;
                                                                                workgroupBarrier();
                                                                                phi_12401_ = select(_e4933_, true, (_e5299_ <= 0.0000005f));
                                                                            } else {
                                                                                phi_12401_ = _e4933_;
                                                                            }
                                                                            let _e5307_ = phi_12401_;
                                                                            phi_36060_ = _e5298_;
                                                                            phi_12403_ = select(_e4927_, _e4999_, _e5303_);
                                                                            phi_12404_ = (_e4929_ * 0.5f);
                                                                            phi_12405_ = select(_e4931_, true, _e5303_);
                                                                            phi_12406_ = _e5307_;
                                                                        }
                                                                        let _e5312_ = phi_36060_;
                                                                        let _e5314_ = phi_12403_;
                                                                        let _e5316_ = phi_12404_;
                                                                        let _e5318_ = phi_12405_;
                                                                        let _e5320_ = phi_12406_;
                                                                        phi_36059_ = _e5312_;
                                                                        phi_10451_ = _e5314_;
                                                                        phi_10453_ = _e5316_;
                                                                        phi_10455_ = _e5318_;
                                                                        phi_10457_ = _e5320_;
                                                                    } else {
                                                                        phi_36059_ = _e4923_;
                                                                        phi_10451_ = f32();
                                                                        phi_10453_ = f32();
                                                                        phi_10455_ = bool();
                                                                        phi_10457_ = bool();
                                                                    }
                                                                    let _e5322_ = phi_36059_;
                                                                    let _e5324_ = phi_10451_;
                                                                    let _e5326_ = phi_10453_;
                                                                    let _e5328_ = phi_10455_;
                                                                    let _e5330_ = phi_10457_;
                                                                    phi_35919_ = _e5322_;
                                                                    phi_10447_ = _e4960_;
                                                                    phi_10450_ = _e5324_;
                                                                    phi_10452_ = _e5326_;
                                                                    phi_10454_ = _e5328_;
                                                                    phi_10456_ = _e5330_;
                                                                    phi_36058_ = _e5322_;
                                                                    if !(_e4966_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e5333_ = phi_36058_;
                                                                phi_36064_ = _e5333_;
                                                                if _e5333_ {
                                                                    break;
                                                                }
                                                                phi_36067_ = _e5333_;
                                                                let _e9831_ = local_45_;
                                                                phi_12407_ = _e9831_;
                                                                let _e9834_ = local_46_;
                                                                phi_12408_ = _e9834_;
                                                            }
                                                            let _e5335_ = phi_36067_;
                                                            let _e5337_ = phi_12407_;
                                                            let _e5339_ = phi_12408_;
                                                            phi_36066_ = _e5335_;
                                                            phi_12409_ = _e5337_;
                                                            phi_12410_ = _e5339_;
                                                        }
                                                        let _e5341_ = phi_36066_;
                                                        let _e5343_ = phi_12409_;
                                                        let _e5345_ = phi_12410_;
                                                        phi_36065_ = _e5341_;
                                                        phi_8500_ = _e5343_;
                                                        phi_8502_ = _e5345_;
                                                    } else {
                                                        phi_36065_ = _e4351_;
                                                        phi_8500_ = f32();
                                                        phi_8502_ = bool();
                                                    }
                                                    let _e5347_ = phi_36065_;
                                                    let _e5349_ = phi_8500_;
                                                    let _e5351_ = phi_8502_;
                                                    phi_35623_ = _e5347_;
                                                    phi_8496_ = _e4370_;
                                                    phi_8499_ = _e5349_;
                                                    phi_8501_ = _e5351_;
                                                    phi_36064_ = _e5347_;
                                                    if !(_e4375_) {
                                                        break;
                                                    }
                                                }
                                                let _e5354_ = phi_36064_;
                                                phi_36579_ = _e5354_;
                                                if _e5354_ {
                                                    break;
                                                }
                                                if (_e158_ == 0u) {
                                                    phi_12424_ = _e4344_;
                                                } else {
                                                    let _e5356_ = (_e4347_ - 1u);
                                                    if (_e5356_ < 1536u) {
                                                    } else {
                                                        phi_36579_ = true;
                                                        break;
                                                    }
                                                    let _e5359_ = scan_scratch[_e5356_];
                                                    phi_12424_ = _e5359_;
                                                }
                                                let _e5361_ = phi_12424_;
                                                if _e1497_ {
                                                    let _e5365_ = (_e4316_ * _e4327_);
                                                    let _e5367_ = bitcast<u32>((_e4309_ * _e4338_));
                                                    let _e5368_ = (_e5367_ & 2139095040u);
                                                    if (_e5368_ == 2139095040u) {
                                                        phi_12454_ = 0u;
                                                    } else {
                                                        if (_e5368_ == 0u) {
                                                            phi_12453_ = (_e5367_ & 2147483648u);
                                                        } else {
                                                            phi_12453_ = _e5367_;
                                                        }
                                                        let _e5373_ = phi_12453_;
                                                        phi_12454_ = _e5373_;
                                                    }
                                                    let _e5375_ = phi_12454_;
                                                    let _e5381_ = (_e4264_ <= 0f);
                                                    let _e5383_ = (1f - _e4226_);
                                                    let _e5384_ = (_e5383_ + select((_e4264_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e5375_)) & 2147483647u))), 0f, _e5381_));
                                                    if (_e5384_ != _e5384_) {
                                                        phi_28311_ = true;
                                                    } else {
                                                        phi_28311_ = (0f >= _e5384_);
                                                    }
                                                    let _e5388_ = phi_28311_;
                                                    let _e5389_ = select(_e5384_, 0f, _e5388_);
                                                    if (_e5389_ != _e5389_) {
                                                        phi_28371_ = true;
                                                    } else {
                                                        phi_28371_ = (1f <= _e5389_);
                                                    }
                                                    let _e5393_ = phi_28371_;
                                                    let _e5394_ = select(_e5389_, 1f, _e5393_);
                                                    let _e5395_ = (_e4207_ * _e5361_);
                                                    let _e5396_ = (1f - _e4283_);
                                                    let _e5397_ = (_e4245_ * _e5396_);
                                                    let _e5399_ = (_e5365_ - (_e5397_ * _e5395_));
                                                    if (_e5399_ != _e5399_) {
                                                        phi_28431_ = true;
                                                    } else {
                                                        phi_28431_ = (-1f >= _e5399_);
                                                    }
                                                    let _e5403_ = phi_28431_;
                                                    let _e5404_ = select(_e5399_, -1f, _e5403_);
                                                    if (_e5404_ != _e5404_) {
                                                        phi_28491_ = true;
                                                    } else {
                                                        phi_28491_ = (1f <= _e5404_);
                                                    }
                                                    let _e5408_ = phi_28491_;
                                                    let _e5409_ = select(_e5404_, 1f, _e5408_);
                                                    let _e5416_ = bitcast<f32>((bitcast<u32>(_e5399_) & 2147483647u));
                                                    if (_e5416_ != _e5416_) {
                                                        phi_28551_ = true;
                                                    } else {
                                                        phi_28551_ = (1000000000000000f <= _e5416_);
                                                    }
                                                    let _e5420_ = phi_28551_;
                                                    let _e5421_ = select(_e5416_, 1000000000000000f, _e5420_);
                                                    let _e5422_ = (_e5421_ * _e5421_);
                                                    let _e5423_ = (_e5422_ + 0.0000010000001f);
                                                    if (_e5423_ != _e5423_) {
                                                        phi_28611_ = true;
                                                    } else {
                                                        phi_28611_ = (0.000000000001f >= _e5423_);
                                                    }
                                                    let _e5427_ = phi_28611_;
                                                    let _e5428_ = select(_e5423_, 0.000000000001f, _e5427_);
                                                    let _e5429_ = (_e5428_ * 0.5f);
                                                    let _e5434_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e5428_) >> bitcast<u32>(1i))));
                                                    let _e5438_ = (_e5434_ * (1.5f - (_e5429_ * (_e5434_ * _e5434_))));
                                                    let _e5442_ = (_e5438_ * (1.5f - (_e5429_ * (_e5438_ * _e5438_))));
                                                    let _e5447_ = (_e5423_ * (_e5442_ * (1.5f - (_e5429_ * (_e5442_ * _e5442_)))));
                                                    let _e5448_ = (_e5447_ + 0.001f);
                                                    if (_e5448_ != _e5448_) {
                                                        phi_28671_ = true;
                                                    } else {
                                                        phi_28671_ = (0.000001f >= _e5448_);
                                                    }
                                                    let _e5452_ = phi_28671_;
                                                    let _e5453_ = select(_e5448_, 0.000001f, _e5452_);
                                                    let _e5456_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5453_)));
                                                    let _e5460_ = (_e5456_ + (_e5456_ * (1f - (_e5453_ * _e5456_))));
                                                    let _e5464_ = (_e5460_ + (_e5460_ * (1f - (_e5453_ * _e5460_))));
                                                    let _e5468_ = (_e5464_ + (_e5464_ * (1f - (_e5453_ * _e5464_))));
                                                    let _e5469_ = (_e5447_ + _e5421_);
                                                    if (_e5469_ != _e5469_) {
                                                        phi_28731_ = true;
                                                    } else {
                                                        phi_28731_ = (0.000001f >= _e5469_);
                                                    }
                                                    let _e5473_ = phi_28731_;
                                                    let _e5474_ = select(_e5469_, 0.000001f, _e5473_);
                                                    let _e5477_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5474_)));
                                                    let _e5481_ = (_e5477_ + (_e5477_ * (1f - (_e5474_ * _e5477_))));
                                                    let _e5485_ = (_e5481_ + (_e5481_ * (1f - (_e5474_ * _e5481_))));
                                                    let _e5495_ = (_e5416_ <= 1000000000000000f);
                                                    let _e5505_ = select(select(select(_e5416_, (0.5f * (_e5421_ + (_e5422_ * _e5468_))), _e5495_), select(-0.0005f, -(((0.0005f * (_e5421_ * (_e5485_ + (_e5485_ * (1f - (_e5474_ * _e5485_)))))) * (1f + (_e5421_ * _e5468_)))), _e5495_), (_e5399_ <= 0f)), 0f, (_e5416_ <= 0f));
                                                    let _e5509_ = (_e4283_ <= 0f);
                                                    let _e5516_ = bitcast<u32>((((1f - (_e4283_ * _e5394_)) * _e5395_) + (_e5394_ * select(((_e5396_ * _e5505_) + (_e4283_ * (_e5409_ * (1f - (0.33333334f * (_e5409_ * _e5409_)))))), _e5505_, _e5509_))));
                                                    let _e5517_ = (_e5516_ & 2139095040u);
                                                    if (_e5517_ == 2139095040u) {
                                                        phi_13196_ = 0u;
                                                    } else {
                                                        if (_e5517_ == 0u) {
                                                            phi_13195_ = (_e5516_ & 2147483648u);
                                                        } else {
                                                            phi_13195_ = _e5516_;
                                                        }
                                                        let _e5522_ = phi_13195_;
                                                        phi_13196_ = _e5522_;
                                                    }
                                                    let _e5524_ = phi_13196_;
                                                    let _e5526_ = bitcast<u32>(_e5365_);
                                                    let _e5527_ = (_e5526_ & 2139095040u);
                                                    if (_e5527_ == 2139095040u) {
                                                        phi_13240_ = true;
                                                    } else {
                                                        phi_13240_ = (_e5527_ == 0u);
                                                    }
                                                    let _e5531_ = phi_13240_;
                                                    let _e5534_ = bitcast<u32>(_e4338_);
                                                    let _e5535_ = (_e5534_ & 2139095040u);
                                                    if (_e5535_ == 2139095040u) {
                                                        phi_13256_ = true;
                                                    } else {
                                                        phi_13256_ = (_e5535_ == 0u);
                                                    }
                                                    let _e5539_ = phi_13256_;
                                                    let _e5542_ = bitcast<u32>(_e5361_);
                                                    let _e5543_ = (_e5542_ & 2139095040u);
                                                    if (_e5543_ == 2139095040u) {
                                                        phi_13272_ = true;
                                                    } else {
                                                        phi_13272_ = (_e5543_ == 0u);
                                                    }
                                                    let _e5547_ = phi_13272_;
                                                    let _e5551_ = bitcast<u32>((_e4309_ * bitcast<f32>(select(_e5534_, 0u, _e5539_))));
                                                    let _e5552_ = (_e5551_ & 2139095040u);
                                                    if (_e5552_ == 2139095040u) {
                                                        phi_13289_ = true;
                                                    } else {
                                                        phi_13289_ = (_e5552_ == 0u);
                                                    }
                                                    let _e5556_ = phi_13289_;
                                                    if _e5381_ {
                                                        phi_13301_ = 0f;
                                                    } else {
                                                        phi_13301_ = (_e4264_ * abs(bitcast<f32>(select(_e5551_, 0u, _e5556_))));
                                                    }
                                                    let _e5562_ = phi_13301_;
                                                    let _e5563_ = (_e5383_ + _e5562_);
                                                    if (_e5563_ <= 0f) {
                                                        phi_13313_ = 0f;
                                                    } else {
                                                        phi_13313_ = select(_e5563_, 1f, (_e5563_ >= 1f));
                                                    }
                                                    let _e5568_ = phi_13313_;
                                                    let _e5571_ = (bitcast<f32>(select(_e5526_, 0u, _e5531_)) - (_e5397_ * (_e4207_ * bitcast<f32>(select(_e5542_, 0u, _e5547_)))));
                                                    let _e5574_ = bitcast<f32>((bitcast<u32>(_e5571_) & 2147483647u));
                                                    if (_e5574_ != _e5574_) {
                                                        phi_28911_ = true;
                                                    } else {
                                                        phi_28911_ = (1000000000000000f <= _e5574_);
                                                    }
                                                    let _e5578_ = phi_28911_;
                                                    let _e5579_ = select(_e5574_, 1000000000000000f, _e5578_);
                                                    let _e5581_ = ((_e5579_ * _e5579_) + 0.0000010000001f);
                                                    if (_e5581_ != _e5581_) {
                                                        phi_28971_ = true;
                                                    } else {
                                                        phi_28971_ = (0.000000000001f >= _e5581_);
                                                    }
                                                    let _e5585_ = phi_28971_;
                                                    let _e5586_ = select(_e5581_, 0.000000000001f, _e5585_);
                                                    let _e5587_ = (_e5586_ * 0.5f);
                                                    let _e5592_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e5586_) >> bitcast<u32>(1i))));
                                                    let _e5596_ = (_e5592_ * (1.5f - (_e5587_ * (_e5592_ * _e5592_))));
                                                    let _e5600_ = (_e5596_ * (1.5f - (_e5587_ * (_e5596_ * _e5596_))));
                                                    let _e5604_ = (_e5600_ * (1.5f - (_e5587_ * (_e5600_ * _e5600_))));
                                                    let _e5606_ = ((_e5581_ * _e5604_) + _e5579_);
                                                    if (_e5606_ != _e5606_) {
                                                        phi_29091_ = true;
                                                    } else {
                                                        phi_29091_ = (0.000001f >= _e5606_);
                                                    }
                                                    let _e5610_ = phi_29091_;
                                                    let _e5611_ = select(_e5606_, 0.000001f, _e5610_);
                                                    let _e5614_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5611_)));
                                                    let _e5618_ = (_e5614_ + (_e5614_ * (1f - (_e5611_ * _e5614_))));
                                                    let _e5622_ = (_e5618_ + (_e5618_ * (1f - (_e5611_ * _e5618_))));
                                                    let _e5627_ = (_e5574_ <= 1000000000000000f);
                                                    let _e5638_ = select(select(select(1f, (0.5f * (1f + (_e5579_ * _e5604_))), _e5627_), select(0f, ((0.00000050000006f * _e5604_) * (_e5622_ + (_e5622_ * (1f - (_e5611_ * _e5622_))))), _e5627_), (_e5571_ <= 0f)), 0.5f, (_e5574_ <= 0f));
                                                    if (_e5638_ != _e5638_) {
                                                        phi_29151_ = true;
                                                    } else {
                                                        phi_29151_ = (0f >= _e5638_);
                                                    }
                                                    let _e5642_ = phi_29151_;
                                                    let _e5643_ = select(_e5638_, 0f, _e5642_);
                                                    if (_e5643_ != _e5643_) {
                                                        phi_29211_ = true;
                                                    } else {
                                                        phi_29211_ = (1f <= _e5643_);
                                                    }
                                                    let _e5647_ = phi_29211_;
                                                    let _e5648_ = select(_e5643_, 1f, _e5647_);
                                                    if (abs(_e5571_) < 1f) {
                                                        phi_13809_ = (1f - (_e5571_ * _e5571_));
                                                    } else {
                                                        phi_13809_ = 0f;
                                                    }
                                                    let _e5654_ = phi_13809_;
                                                    if _e5509_ {
                                                        phi_13817_ = _e5648_;
                                                    } else {
                                                        phi_13817_ = ((_e5396_ * _e5648_) + (_e4283_ * _e5654_));
                                                    }
                                                    let _e5659_ = phi_13817_;
                                                    let _e5666_ = (_e4207_ * ((1f - (_e4283_ * _e5568_)) - (((_e5568_ * _e4245_) * _e5396_) * _e5659_)));
                                                    let _e5667_ = bitcast<u32>(_e5666_);
                                                    let _e5668_ = (_e5667_ & 2139095040u);
                                                    let _e5669_ = (_e5668_ == 2139095040u);
                                                    if _e5669_ {
                                                        if ((_e5667_ & 2147483647u) > 2139095040u) {
                                                            phi_13874_ = f32();
                                                            phi_13875_ = true;
                                                        } else {
                                                            let _e5684_ = ((_e5667_ & 2147483648u) == 0u);
                                                            if _e5684_ {
                                                                phi_13872_ = 0.999995f;
                                                            } else {
                                                                phi_13872_ = f32();
                                                            }
                                                            let _e5686_ = phi_13872_;
                                                            phi_13874_ = _e5686_;
                                                            phi_13875_ = select(true, false, _e5684_);
                                                        }
                                                        let _e5689_ = phi_13874_;
                                                        let _e5691_ = phi_13875_;
                                                        if _e5691_ {
                                                            phi_13895_ = -0.999995f;
                                                        } else {
                                                            phi_13895_ = _e5689_;
                                                        }
                                                        let _e5693_ = phi_13895_;
                                                        phi_13954_ = _e5693_;
                                                    } else {
                                                        if (_e5666_ < -0.999995f) {
                                                            phi_13953_ = -0.999995f;
                                                        } else {
                                                            if (_e5666_ > 0.999995f) {
                                                                phi_13952_ = 0.999995f;
                                                            } else {
                                                                if _e5669_ {
                                                                    phi_13946_ = true;
                                                                } else {
                                                                    phi_13946_ = (_e5668_ == 0u);
                                                                }
                                                                let _e5674_ = phi_13946_;
                                                                phi_13952_ = bitcast<f32>(select(_e5667_, 0u, _e5674_));
                                                            }
                                                            let _e5678_ = phi_13952_;
                                                            phi_13953_ = _e5678_;
                                                        }
                                                        let _e5680_ = phi_13953_;
                                                        phi_13954_ = _e5680_;
                                                    }
                                                    let _e5695_ = phi_13954_;
                                                    phi_13957_ = type_28_(bitcast<f32>(_e5524_), _e5695_);
                                                } else {
                                                    let _e5363_ = local_37_;
                                                    phi_13957_ = type_28_(_e5363_, 1f);
                                                }
                                                let _e5698_ = phi_13957_;
                                                if _e1497_ {
                                                    if ((bitcast<u32>(_e5698_.member) & 2139095040u) != 2139095040u) {
                                                        let _e5704_ = local_38_;
                                                        let _e5707_ = ((bitcast<u32>(_e5704_) & 2139095040u) != 2139095040u);
                                                        if _e5707_ {
                                                            let _e5708_ = abs(_e5698_.member);
                                                            let _e5710_ = local_39_;
                                                            let _e5711_ = abs(_e5710_);
                                                            let _e5713_ = select(_e5711_, _e5708_, (_e5708_ > _e5711_));
                                                            let _e5715_ = select(1f, _e5713_, (_e5713_ > 1f));
                                                            let _e5717_ = local_40_;
                                                            if ((bitcast<u32>(_e5715_) & 2139095040u) != 2139095040u) {
                                                                let _e5723_ = (_e5715_ <= 0f);
                                                                if _e5723_ {
                                                                    phi_14080_ = f32();
                                                                } else {
                                                                    if (_e5715_ < 0.00001f) {
                                                                        let _e5755_ = (select(0.000000000000000000000001f, _e5715_, (_e5715_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                        if (_e5755_ != _e5755_) {
                                                                            phi_29331_ = true;
                                                                        } else {
                                                                            phi_29331_ = (0.000001f >= _e5755_);
                                                                        }
                                                                        let _e5759_ = phi_29331_;
                                                                        let _e5760_ = select(_e5755_, 0.000001f, _e5759_);
                                                                        let _e5763_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5760_)));
                                                                        let _e5767_ = (_e5763_ + (_e5763_ * (1f - (_e5760_ * _e5763_))));
                                                                        let _e5771_ = (_e5767_ + (_e5767_ * (1f - (_e5760_ * _e5767_))));
                                                                        let _e5777_ = bitcast<u32>(((_e5771_ + (_e5771_ * (1f - (_e5760_ * _e5771_)))) * 18446744000000000000f));
                                                                        let _e5778_ = (_e5777_ & 2139095040u);
                                                                        if (_e5778_ == 2139095040u) {
                                                                            phi_14038_ = true;
                                                                        } else {
                                                                            phi_14038_ = (_e5778_ == 0u);
                                                                        }
                                                                        let _e5782_ = phi_14038_;
                                                                        phi_14079_ = bitcast<f32>(select(_e5777_, 0u, _e5782_));
                                                                    } else {
                                                                        if (_e5715_ != _e5715_) {
                                                                            phi_29271_ = true;
                                                                        } else {
                                                                            phi_29271_ = (0.000001f >= _e5715_);
                                                                        }
                                                                        let _e5728_ = phi_29271_;
                                                                        let _e5729_ = select(_e5715_, 0.000001f, _e5728_);
                                                                        let _e5732_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5729_)));
                                                                        let _e5736_ = (_e5732_ + (_e5732_ * (1f - (_e5729_ * _e5732_))));
                                                                        let _e5740_ = (_e5736_ + (_e5736_ * (1f - (_e5729_ * _e5736_))));
                                                                        let _e5745_ = bitcast<u32>((_e5740_ + (_e5740_ * (1f - (_e5729_ * _e5740_)))));
                                                                        let _e5746_ = (_e5745_ & 2139095040u);
                                                                        if (_e5746_ == 2139095040u) {
                                                                            phi_14073_ = true;
                                                                        } else {
                                                                            phi_14073_ = (_e5746_ == 0u);
                                                                        }
                                                                        let _e5750_ = phi_14073_;
                                                                        phi_14079_ = bitcast<f32>(select(_e5745_, 0u, _e5750_));
                                                                    }
                                                                    let _e5786_ = phi_14079_;
                                                                    phi_14080_ = _e5786_;
                                                                }
                                                                let _e5788_ = phi_14080_;
                                                                phi_14081_ = _e5788_;
                                                                phi_14082_ = _e5723_;
                                                            } else {
                                                                phi_14081_ = f32();
                                                                phi_14082_ = true;
                                                            }
                                                            let _e5790_ = phi_14081_;
                                                            let _e5792_ = phi_14082_;
                                                            phi_14088_ = (abs((_e5698_.member - _e5717_)) * select(_e5790_, 0f, _e5792_));
                                                        } else {
                                                            phi_14088_ = f32();
                                                        }
                                                        let _e5796_ = phi_14088_;
                                                        phi_14090_ = _e5796_;
                                                        phi_14091_ = select(true, false, _e5707_);
                                                    } else {
                                                        phi_14090_ = f32();
                                                        phi_14091_ = true;
                                                    }
                                                    let _e5799_ = phi_14090_;
                                                    let _e5801_ = phi_14091_;
                                                    phi_14096_ = select(_e5799_, 1000000000f, _e5801_);
                                                } else {
                                                    phi_14096_ = 0f;
                                                }
                                                let _e5804_ = phi_14096_;
                                                let _e5805_ = (_e157_ < 128u);
                                                if _e5805_ {
                                                    let _e5806_ = (32u + _e158_);
                                                    if (_e5806_ < 256u) {
                                                    } else {
                                                        phi_36579_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e5806_] = _e5804_;
                                                }
                                                workgroupBarrier();
                                                if (_e157_ >= 128u) {
                                                    let _e5810_ = (32u + _e158_);
                                                    let _e5811_ = (_e5810_ < 256u);
                                                    if _e5811_ {
                                                    } else {
                                                        phi_36579_ = true;
                                                        break;
                                                    }
                                                    let _e5813_ = aux_scratch[_e5810_];
                                                    if _e5811_ {
                                                    } else {
                                                        phi_36579_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e5810_] = select(_e5813_, _e5804_, (_e5804_ > _e5813_));
                                                }
                                                workgroupBarrier();
                                                phi_14134_ = 64u;
                                                loop {
                                                    let _e5817_ = phi_14134_;
                                                    let _e5818_ = (_e5817_ > 0u);
                                                    if _e5818_ {
                                                        if (_e158_ < _e5817_) {
                                                            if _e5805_ {
                                                                let _e5820_ = (32u + _e158_);
                                                                let _e5821_ = (_e5820_ < 256u);
                                                                if _e5821_ {
                                                                } else {
                                                                    phi_36068_ = true;
                                                                    break;
                                                                }
                                                                let _e5823_ = aux_scratch[_e5820_];
                                                                let _e5824_ = (_e5820_ + _e5817_);
                                                                if (_e5824_ < 256u) {
                                                                } else {
                                                                    phi_36068_ = true;
                                                                    break;
                                                                }
                                                                let _e5827_ = aux_scratch[_e5824_];
                                                                if _e5821_ {
                                                                } else {
                                                                    phi_36068_ = true;
                                                                    break;
                                                                }
                                                                aux_scratch[_e5820_] = select(_e5823_, _e5827_, (_e5827_ > _e5823_));
                                                            }
                                                        }
                                                        workgroupBarrier();
                                                        phi_14135_ = (_e5817_ >> bitcast<u32>(1i));
                                                    } else {
                                                        phi_14135_ = u32();
                                                    }
                                                    let _e5833_ = phi_14135_;
                                                    phi_14134_ = _e5833_;
                                                    phi_36068_ = _e5354_;
                                                    if !(_e5818_) {
                                                        break;
                                                    }
                                                }
                                                let _e5836_ = phi_36068_;
                                                phi_36579_ = _e5836_;
                                                if _e5836_ {
                                                    break;
                                                }
                                                let _e5838_ = aux_scratch[32];
                                                workgroupBarrier();
                                                if (_e5838_ > 0.0000005f) {
                                                    if _e1098_ {
                                                        let _e5884_ = (16u + _e4141_.member_1_);
                                                        let _e5885_ = (_e5884_ < 256u);
                                                        if _e5885_ {
                                                        } else {
                                                            phi_36579_ = true;
                                                            break;
                                                        }
                                                        let _e5887_ = aux_scratch[_e5884_];
                                                        let _e5888_ = (20u + _e4141_.member_1_);
                                                        let _e5889_ = (_e5888_ < 256u);
                                                        if _e5889_ {
                                                        } else {
                                                            phi_36579_ = true;
                                                            break;
                                                        }
                                                        let _e5891_ = aux_scratch[_e5888_];
                                                        phi_14221_ = type_15_(0u, _e1495_);
                                                        phi_14224_ = _e5891_;
                                                        phi_14226_ = _e5887_;
                                                        loop {
                                                            let _e5894_ = phi_14221_;
                                                            let _e5896_ = phi_14224_;
                                                            let _e5898_ = phi_14226_;
                                                            local_41_ = _e5898_;
                                                            local_42_ = _e5896_;
                                                            if (_e5894_.member < _e5894_.member_1_) {
                                                                phi_14222_ = type_15_((_e5894_.member + 1u), _e5894_.member_1_);
                                                                phi_14249_ = type_15_(1u, _e5894_.member);
                                                            } else {
                                                                phi_14222_ = _e5894_;
                                                                phi_14249_ = type_15_(0u, u32());
                                                            }
                                                            let _e5911_ = phi_14222_;
                                                            let _e5913_ = phi_14249_;
                                                            let _e5917_ = (bitcast<i32>(_e5913_.member) != 0i);
                                                            if _e5917_ {
                                                                let _e5919_ = (_e4141_.member_1_ * 4u);
                                                                let _e5921_ = ((_e5913_.member_1_ & 127u) & 127u);
                                                                let _e5922_ = ((_e4141_.member_1_ * 512u) + _e5921_);
                                                                if (_e5922_ < 2048u) {
                                                                } else {
                                                                    phi_36303_ = true;
                                                                    break;
                                                                }
                                                                let _e5925_ = lane_buf[_e5922_];
                                                                let _e5928_ = (((_e5919_ + 1u) * 128u) + _e5921_);
                                                                let _e5929_ = (_e5928_ < 2048u);
                                                                if _e5929_ {
                                                                } else {
                                                                    phi_36303_ = true;
                                                                    break;
                                                                }
                                                                let _e5931_ = lane_buf[_e5928_];
                                                                let _e5934_ = (((_e5919_ + 2u) * 128u) + _e5921_);
                                                                if (_e5934_ < 2048u) {
                                                                } else {
                                                                    phi_36303_ = true;
                                                                    break;
                                                                }
                                                                let _e5937_ = lane_buf[_e5934_];
                                                                let _e5940_ = (((_e5919_ + 3u) * 128u) + _e5921_);
                                                                let _e5941_ = (_e5940_ < 2048u);
                                                                if _e5941_ {
                                                                } else {
                                                                    phi_36303_ = true;
                                                                    break;
                                                                }
                                                                let _e5943_ = lane_buf[_e5940_];
                                                                let _e5946_ = bitcast<u32>((_e4309_ * _e5931_));
                                                                let _e5947_ = (_e5946_ & 2139095040u);
                                                                if (_e5947_ == 2139095040u) {
                                                                    phi_14320_ = 0u;
                                                                } else {
                                                                    if (_e5947_ == 0u) {
                                                                        phi_14319_ = (_e5946_ & 2147483648u);
                                                                    } else {
                                                                        phi_14319_ = _e5946_;
                                                                    }
                                                                    let _e5952_ = phi_14319_;
                                                                    phi_14320_ = _e5952_;
                                                                }
                                                                let _e5954_ = phi_14320_;
                                                                let _e5960_ = (_e4264_ <= 0f);
                                                                let _e5962_ = (1f - _e4226_);
                                                                let _e5963_ = (_e5962_ + select((_e4264_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e5954_)) & 2147483647u))), 0f, _e5960_));
                                                                if (_e5963_ != _e5963_) {
                                                                    phi_29391_ = true;
                                                                } else {
                                                                    phi_29391_ = (0f >= _e5963_);
                                                                }
                                                                let _e5967_ = phi_29391_;
                                                                let _e5968_ = select(_e5963_, 0f, _e5967_);
                                                                if (_e5968_ != _e5968_) {
                                                                    phi_29451_ = true;
                                                                } else {
                                                                    phi_29451_ = (1f <= _e5968_);
                                                                }
                                                                let _e5972_ = phi_29451_;
                                                                let _e5973_ = select(_e5968_, 1f, _e5972_);
                                                                let _e5974_ = (_e4207_ * _e5898_);
                                                                let _e5975_ = (1f - _e4283_);
                                                                let _e5976_ = (_e4245_ * _e5975_);
                                                                let _e5978_ = ((_e4316_ * _e5925_) - (_e5976_ * _e5974_));
                                                                if (_e5978_ != _e5978_) {
                                                                    phi_29511_ = true;
                                                                } else {
                                                                    phi_29511_ = (-1f >= _e5978_);
                                                                }
                                                                let _e5982_ = phi_29511_;
                                                                let _e5983_ = select(_e5978_, -1f, _e5982_);
                                                                if (_e5983_ != _e5983_) {
                                                                    phi_29571_ = true;
                                                                } else {
                                                                    phi_29571_ = (1f <= _e5983_);
                                                                }
                                                                let _e5987_ = phi_29571_;
                                                                let _e5988_ = select(_e5983_, 1f, _e5987_);
                                                                let _e5995_ = bitcast<f32>((bitcast<u32>(_e5978_) & 2147483647u));
                                                                if (_e5995_ != _e5995_) {
                                                                    phi_29631_ = true;
                                                                } else {
                                                                    phi_29631_ = (1000000000000000f <= _e5995_);
                                                                }
                                                                let _e5999_ = phi_29631_;
                                                                let _e6000_ = select(_e5995_, 1000000000000000f, _e5999_);
                                                                let _e6001_ = (_e6000_ * _e6000_);
                                                                let _e6002_ = (_e6001_ + 0.0000010000001f);
                                                                if (_e6002_ != _e6002_) {
                                                                    phi_29691_ = true;
                                                                } else {
                                                                    phi_29691_ = (0.000000000001f >= _e6002_);
                                                                }
                                                                let _e6006_ = phi_29691_;
                                                                let _e6007_ = select(_e6002_, 0.000000000001f, _e6006_);
                                                                let _e6008_ = (_e6007_ * 0.5f);
                                                                let _e6013_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e6007_) >> bitcast<u32>(1i))));
                                                                let _e6017_ = (_e6013_ * (1.5f - (_e6008_ * (_e6013_ * _e6013_))));
                                                                let _e6021_ = (_e6017_ * (1.5f - (_e6008_ * (_e6017_ * _e6017_))));
                                                                let _e6026_ = (_e6002_ * (_e6021_ * (1.5f - (_e6008_ * (_e6021_ * _e6021_)))));
                                                                let _e6027_ = (_e6026_ + 0.001f);
                                                                if (_e6027_ != _e6027_) {
                                                                    phi_29751_ = true;
                                                                } else {
                                                                    phi_29751_ = (0.000001f >= _e6027_);
                                                                }
                                                                let _e6031_ = phi_29751_;
                                                                let _e6032_ = select(_e6027_, 0.000001f, _e6031_);
                                                                let _e6035_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6032_)));
                                                                let _e6039_ = (_e6035_ + (_e6035_ * (1f - (_e6032_ * _e6035_))));
                                                                let _e6043_ = (_e6039_ + (_e6039_ * (1f - (_e6032_ * _e6039_))));
                                                                let _e6047_ = (_e6043_ + (_e6043_ * (1f - (_e6032_ * _e6043_))));
                                                                let _e6048_ = (_e6026_ + _e6000_);
                                                                if (_e6048_ != _e6048_) {
                                                                    phi_29811_ = true;
                                                                } else {
                                                                    phi_29811_ = (0.000001f >= _e6048_);
                                                                }
                                                                let _e6052_ = phi_29811_;
                                                                let _e6053_ = select(_e6048_, 0.000001f, _e6052_);
                                                                let _e6056_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6053_)));
                                                                let _e6060_ = (_e6056_ + (_e6056_ * (1f - (_e6053_ * _e6056_))));
                                                                let _e6064_ = (_e6060_ + (_e6060_ * (1f - (_e6053_ * _e6060_))));
                                                                let _e6074_ = (_e5995_ <= 1000000000000000f);
                                                                let _e6084_ = select(select(select(_e5995_, (0.5f * (_e6000_ + (_e6001_ * _e6047_))), _e6074_), select(-0.0005f, -(((0.0005f * (_e6000_ * (_e6064_ + (_e6064_ * (1f - (_e6053_ * _e6064_)))))) * (1f + (_e6000_ * _e6047_)))), _e6074_), (_e5978_ <= 0f)), 0f, (_e5995_ <= 0f));
                                                                let _e6088_ = (_e4283_ <= 0f);
                                                                let _e6095_ = bitcast<u32>((((1f - (_e4283_ * _e5973_)) * _e5974_) + (_e5973_ * select(((_e5975_ * _e6084_) + (_e4283_ * (_e5988_ * (1f - (0.33333334f * (_e5988_ * _e5988_)))))), _e6084_, _e6088_))));
                                                                let _e6096_ = (_e6095_ & 2139095040u);
                                                                if (_e6096_ == 2139095040u) {
                                                                    phi_15062_ = 0u;
                                                                } else {
                                                                    if (_e6096_ == 0u) {
                                                                        phi_15061_ = (_e6095_ & 2147483648u);
                                                                    } else {
                                                                        phi_15061_ = _e6095_;
                                                                    }
                                                                    let _e6101_ = phi_15061_;
                                                                    phi_15062_ = _e6101_;
                                                                }
                                                                let _e6103_ = phi_15062_;
                                                                let _e6104_ = bitcast<f32>(_e6103_);
                                                                let _e6107_ = bitcast<u32>((_e4309_ * _e5943_));
                                                                let _e6108_ = (_e6107_ & 2139095040u);
                                                                if (_e6108_ == 2139095040u) {
                                                                    phi_15124_ = 0u;
                                                                } else {
                                                                    if (_e6108_ == 0u) {
                                                                        phi_15123_ = (_e6107_ & 2147483648u);
                                                                    } else {
                                                                        phi_15123_ = _e6107_;
                                                                    }
                                                                    let _e6113_ = phi_15123_;
                                                                    phi_15124_ = _e6113_;
                                                                }
                                                                let _e6115_ = phi_15124_;
                                                                let _e6122_ = (_e5962_ + select((_e4264_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e6115_)) & 2147483647u))), 0f, _e5960_));
                                                                if (_e6122_ != _e6122_) {
                                                                    phi_29991_ = true;
                                                                } else {
                                                                    phi_29991_ = (0f >= _e6122_);
                                                                }
                                                                let _e6126_ = phi_29991_;
                                                                let _e6127_ = select(_e6122_, 0f, _e6126_);
                                                                if (_e6127_ != _e6127_) {
                                                                    phi_30051_ = true;
                                                                } else {
                                                                    phi_30051_ = (1f <= _e6127_);
                                                                }
                                                                let _e6131_ = phi_30051_;
                                                                let _e6132_ = select(_e6127_, 1f, _e6131_);
                                                                let _e6133_ = (_e4207_ * _e5896_);
                                                                let _e6135_ = ((_e4316_ * _e5937_) - (_e5976_ * _e6133_));
                                                                if (_e6135_ != _e6135_) {
                                                                    phi_30111_ = true;
                                                                } else {
                                                                    phi_30111_ = (-1f >= _e6135_);
                                                                }
                                                                let _e6139_ = phi_30111_;
                                                                let _e6140_ = select(_e6135_, -1f, _e6139_);
                                                                if (_e6140_ != _e6140_) {
                                                                    phi_30171_ = true;
                                                                } else {
                                                                    phi_30171_ = (1f <= _e6140_);
                                                                }
                                                                let _e6144_ = phi_30171_;
                                                                let _e6145_ = select(_e6140_, 1f, _e6144_);
                                                                let _e6152_ = bitcast<f32>((bitcast<u32>(_e6135_) & 2147483647u));
                                                                if (_e6152_ != _e6152_) {
                                                                    phi_30231_ = true;
                                                                } else {
                                                                    phi_30231_ = (1000000000000000f <= _e6152_);
                                                                }
                                                                let _e6156_ = phi_30231_;
                                                                let _e6157_ = select(_e6152_, 1000000000000000f, _e6156_);
                                                                let _e6158_ = (_e6157_ * _e6157_);
                                                                let _e6159_ = (_e6158_ + 0.0000010000001f);
                                                                if (_e6159_ != _e6159_) {
                                                                    phi_30291_ = true;
                                                                } else {
                                                                    phi_30291_ = (0.000000000001f >= _e6159_);
                                                                }
                                                                let _e6163_ = phi_30291_;
                                                                let _e6164_ = select(_e6159_, 0.000000000001f, _e6163_);
                                                                let _e6165_ = (_e6164_ * 0.5f);
                                                                let _e6170_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e6164_) >> bitcast<u32>(1i))));
                                                                let _e6174_ = (_e6170_ * (1.5f - (_e6165_ * (_e6170_ * _e6170_))));
                                                                let _e6178_ = (_e6174_ * (1.5f - (_e6165_ * (_e6174_ * _e6174_))));
                                                                let _e6183_ = (_e6159_ * (_e6178_ * (1.5f - (_e6165_ * (_e6178_ * _e6178_)))));
                                                                let _e6184_ = (_e6183_ + 0.001f);
                                                                if (_e6184_ != _e6184_) {
                                                                    phi_30351_ = true;
                                                                } else {
                                                                    phi_30351_ = (0.000001f >= _e6184_);
                                                                }
                                                                let _e6188_ = phi_30351_;
                                                                let _e6189_ = select(_e6184_, 0.000001f, _e6188_);
                                                                let _e6192_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6189_)));
                                                                let _e6196_ = (_e6192_ + (_e6192_ * (1f - (_e6189_ * _e6192_))));
                                                                let _e6200_ = (_e6196_ + (_e6196_ * (1f - (_e6189_ * _e6196_))));
                                                                let _e6204_ = (_e6200_ + (_e6200_ * (1f - (_e6189_ * _e6200_))));
                                                                let _e6205_ = (_e6183_ + _e6157_);
                                                                if (_e6205_ != _e6205_) {
                                                                    phi_30411_ = true;
                                                                } else {
                                                                    phi_30411_ = (0.000001f >= _e6205_);
                                                                }
                                                                let _e6209_ = phi_30411_;
                                                                let _e6210_ = select(_e6205_, 0.000001f, _e6209_);
                                                                let _e6213_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6210_)));
                                                                let _e6217_ = (_e6213_ + (_e6213_ * (1f - (_e6210_ * _e6213_))));
                                                                let _e6221_ = (_e6217_ + (_e6217_ * (1f - (_e6210_ * _e6217_))));
                                                                let _e6231_ = (_e6152_ <= 1000000000000000f);
                                                                let _e6241_ = select(select(select(_e6152_, (0.5f * (_e6157_ + (_e6158_ * _e6204_))), _e6231_), select(-0.0005f, -(((0.0005f * (_e6157_ * (_e6221_ + (_e6221_ * (1f - (_e6210_ * _e6221_)))))) * (1f + (_e6157_ * _e6204_)))), _e6231_), (_e6135_ <= 0f)), 0f, (_e6152_ <= 0f));
                                                                let _e6251_ = bitcast<u32>((((1f - (_e4283_ * _e6132_)) * _e6133_) + (_e6132_ * select(((_e5975_ * _e6241_) + (_e4283_ * (_e6145_ * (1f - (0.33333334f * (_e6145_ * _e6145_)))))), _e6241_, _e6088_))));
                                                                let _e6252_ = (_e6251_ & 2139095040u);
                                                                if (_e6252_ == 2139095040u) {
                                                                    phi_15866_ = 0u;
                                                                } else {
                                                                    if (_e6252_ == 0u) {
                                                                        phi_15865_ = (_e6251_ & 2147483648u);
                                                                    } else {
                                                                        phi_15865_ = _e6251_;
                                                                    }
                                                                    let _e6257_ = phi_15865_;
                                                                    phi_15866_ = _e6257_;
                                                                }
                                                                let _e6259_ = phi_15866_;
                                                                let _e6260_ = bitcast<f32>(_e6259_);
                                                                let _e6262_ = (0.5f * (_e6104_ + _e6260_));
                                                                if _e5929_ {
                                                                } else {
                                                                    phi_36303_ = true;
                                                                    break;
                                                                }
                                                                lane_buf[_e5928_] = (_e6104_ + (_e4302_ * (_e6262_ - _e6104_)));
                                                                if _e5941_ {
                                                                } else {
                                                                    phi_36303_ = true;
                                                                    break;
                                                                }
                                                                lane_buf[_e5940_] = (_e6260_ + (_e4302_ * (_e6262_ - _e6260_)));
                                                                phi_14225_ = _e6260_;
                                                                phi_14227_ = _e6104_;
                                                            } else {
                                                                phi_14225_ = f32();
                                                                phi_14227_ = f32();
                                                            }
                                                            let _e6270_ = phi_14225_;
                                                            let _e6272_ = phi_14227_;
                                                            phi_14221_ = _e5911_;
                                                            phi_14224_ = _e6270_;
                                                            phi_14226_ = _e6272_;
                                                            phi_36303_ = _e5836_;
                                                            if !(_e5917_) {
                                                                break;
                                                            }
                                                        }
                                                        let _e6275_ = phi_36303_;
                                                        phi_36579_ = _e6275_;
                                                        if _e6275_ {
                                                            break;
                                                        }
                                                        if _e5885_ {
                                                        } else {
                                                            phi_36579_ = true;
                                                            break;
                                                        }
                                                        let _e6277_ = local_41_;
                                                        aux_scratch[_e5884_] = _e6277_;
                                                        if _e5889_ {
                                                        } else {
                                                            phi_36579_ = true;
                                                            break;
                                                        }
                                                        let _e6279_ = local_42_;
                                                        aux_scratch[_e5888_] = _e6279_;
                                                        phi_36586_ = _e6275_;
                                                    } else {
                                                        phi_36586_ = _e5836_;
                                                    }
                                                    let _e6281_ = phi_36586_;
                                                    workgroupBarrier();
                                                    phi_36581_ = _e6281_;
                                                } else {
                                                    workgroupBarrier();
                                                    if _e5805_ {
                                                        if _e1497_ {
                                                            let _e5840_ = (512u + _e158_);
                                                            if (_e5840_ < 1536u) {
                                                            } else {
                                                                phi_36579_ = true;
                                                                break;
                                                            }
                                                            let _e5843_ = scan_scratch[_e5840_];
                                                            let _e5844_ = (640u + _e158_);
                                                            if (_e5844_ < 1536u) {
                                                            } else {
                                                                phi_36579_ = true;
                                                                break;
                                                            }
                                                            let _e5847_ = scan_scratch[_e5844_];
                                                            let _e5849_ = (0.5f * (_e5843_ + _e5847_));
                                                            let _e5856_ = (_e4141_.member_1_ * 4u);
                                                            let _e5859_ = (_e158_ & 127u);
                                                            let _e5860_ = (((_e5856_ + 1u) * 128u) + _e5859_);
                                                            if (_e5860_ < 2048u) {
                                                            } else {
                                                                phi_36579_ = true;
                                                                break;
                                                            }
                                                            lane_buf[_e5860_] = (_e5843_ + (_e4302_ * (_e5849_ - _e5843_)));
                                                            let _e5865_ = (((_e5856_ + 3u) * 128u) + _e5859_);
                                                            if (_e5865_ < 2048u) {
                                                            } else {
                                                                phi_36579_ = true;
                                                                break;
                                                            }
                                                            lane_buf[_e5865_] = (_e5847_ + (_e4302_ * (_e5849_ - _e5847_)));
                                                        }
                                                    }
                                                    if _e1098_ {
                                                        if _e1496_ {
                                                            let _e5869_ = ((_e1495_ - 1u) & 127u);
                                                            let _e5870_ = (512u + _e5869_);
                                                            if (_e5870_ < 1536u) {
                                                            } else {
                                                                phi_36579_ = true;
                                                                break;
                                                            }
                                                            let _e5873_ = scan_scratch[_e5870_];
                                                            let _e5874_ = (16u + _e4141_.member_1_);
                                                            if (_e5874_ < 256u) {
                                                            } else {
                                                                phi_36579_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e5874_] = _e5873_;
                                                            let _e5877_ = (640u + _e5869_);
                                                            if (_e5877_ < 1536u) {
                                                            } else {
                                                                phi_36579_ = true;
                                                                break;
                                                            }
                                                            let _e5880_ = scan_scratch[_e5877_];
                                                            let _e5881_ = (20u + _e4141_.member_1_);
                                                            if (_e5881_ < 256u) {
                                                            } else {
                                                                phi_36579_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e5881_] = _e5880_;
                                                        }
                                                    }
                                                    workgroupBarrier();
                                                    phi_36581_ = _e5836_;
                                                }
                                                let _e6283_ = phi_36581_;
                                                phi_36580_ = _e6283_;
                                            } else {
                                                phi_36580_ = _e4124_;
                                            }
                                            let _e6285_ = phi_36580_;
                                            phi_35650_ = _e6285_;
                                            phi_7873_ = _e4139_;
                                            phi_36579_ = _e6285_;
                                            if !(_e4145_) {
                                                break;
                                            }
                                        }
                                        let _e6288_ = phi_36579_;
                                        phi_37452_ = _e6288_;
                                        if _e6288_ {
                                            break;
                                        }
                                        phi_36591_ = _e6288_;
                                    } else {
                                        if (_e157_ < 4u) {
                                            let _e3538_ = (_e157_ & 3u);
                                            switch bitcast<i32>(_e3538_) {
                                                case 0: {
                                                    phi_16108_ = _e370_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16108_ = _e370_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16108_ = _e370_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16108_ = _e370_.w;
                                                    break;
                                                }
                                            }
                                            let _e3545_ = phi_16108_;
                                            switch bitcast<i32>(_e3538_) {
                                                case 0: {
                                                    phi_16119_ = _e377_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16119_ = _e377_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16119_ = _e377_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16119_ = _e377_.w;
                                                    break;
                                                }
                                            }
                                            let _e3552_ = phi_16119_;
                                            switch bitcast<i32>(_e3538_) {
                                                case 0: {
                                                    phi_16130_ = _e384_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16130_ = _e384_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16130_ = _e384_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16130_ = _e384_.w;
                                                    break;
                                                }
                                            }
                                            let _e3559_ = phi_16130_;
                                            switch bitcast<i32>(_e3538_) {
                                                case 0: {
                                                    phi_16141_ = _e391_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16141_ = _e391_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16141_ = _e391_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16141_ = _e391_.w;
                                                    break;
                                                }
                                            }
                                            let _e3566_ = phi_16141_;
                                            switch bitcast<i32>(_e3538_) {
                                                case 0: {
                                                    phi_16152_ = _e398_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16152_ = _e398_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16152_ = _e398_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16152_ = _e398_.w;
                                                    break;
                                                }
                                            }
                                            let _e3573_ = phi_16152_;
                                            switch bitcast<i32>(_e3538_) {
                                                case 0: {
                                                    phi_16163_ = _e405_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16163_ = _e405_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16163_ = _e405_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16163_ = _e405_.w;
                                                    break;
                                                }
                                            }
                                            let _e3580_ = phi_16163_;
                                            let _e3581_ = bitcast<u32>(_e3545_);
                                            let _e3582_ = (_e3581_ & 2139095040u);
                                            if (_e3582_ != 2139095040u) {
                                                if (_e3545_ <= 0f) {
                                                    phi_16266_ = 0f;
                                                } else {
                                                    if (_e3545_ >= 0.999995f) {
                                                        phi_16265_ = 0.999995f;
                                                    } else {
                                                        if (_e3582_ == 2139095040u) {
                                                            phi_16259_ = true;
                                                        } else {
                                                            phi_16259_ = (_e3582_ == 0u);
                                                        }
                                                        let _e3591_ = phi_16259_;
                                                        phi_16265_ = bitcast<f32>(select(_e3581_, 0u, _e3591_));
                                                    }
                                                    let _e3595_ = phi_16265_;
                                                    phi_16266_ = _e3595_;
                                                }
                                                let _e3597_ = phi_16266_;
                                                phi_16273_ = _e3597_;
                                            } else {
                                                phi_16273_ = select(0f, 0.999995f, (_e3581_ == 2139095040u));
                                            }
                                            let _e3599_ = phi_16273_;
                                            let _e3600_ = bitcast<u32>(_e3552_);
                                            let _e3601_ = (_e3600_ & 2139095040u);
                                            if (_e3601_ != 2139095040u) {
                                                if (_e3552_ <= 0f) {
                                                    phi_16305_ = 0f;
                                                } else {
                                                    if (_e3552_ >= 1f) {
                                                        phi_16304_ = 1f;
                                                    } else {
                                                        if (_e3601_ == 2139095040u) {
                                                            phi_16298_ = true;
                                                        } else {
                                                            phi_16298_ = (_e3601_ == 0u);
                                                        }
                                                        let _e3610_ = phi_16298_;
                                                        phi_16304_ = bitcast<f32>(select(_e3600_, 0u, _e3610_));
                                                    }
                                                    let _e3614_ = phi_16304_;
                                                    phi_16305_ = _e3614_;
                                                }
                                                let _e3616_ = phi_16305_;
                                                phi_16312_ = _e3616_;
                                            } else {
                                                phi_16312_ = select(0f, 1f, (_e3600_ == 2139095040u));
                                            }
                                            let _e3618_ = phi_16312_;
                                            let _e3619_ = bitcast<u32>(_e3559_);
                                            let _e3620_ = (_e3619_ & 2139095040u);
                                            if (_e3620_ != 2139095040u) {
                                                if (_e3559_ <= 0f) {
                                                    phi_16344_ = 0f;
                                                } else {
                                                    if (_e3559_ >= 1f) {
                                                        phi_16343_ = 1f;
                                                    } else {
                                                        if (_e3620_ == 2139095040u) {
                                                            phi_16337_ = true;
                                                        } else {
                                                            phi_16337_ = (_e3620_ == 0u);
                                                        }
                                                        let _e3629_ = phi_16337_;
                                                        phi_16343_ = bitcast<f32>(select(_e3619_, 0u, _e3629_));
                                                    }
                                                    let _e3633_ = phi_16343_;
                                                    phi_16344_ = _e3633_;
                                                }
                                                let _e3635_ = phi_16344_;
                                                phi_16351_ = _e3635_;
                                            } else {
                                                phi_16351_ = select(0f, 1f, (_e3619_ == 2139095040u));
                                            }
                                            let _e3637_ = phi_16351_;
                                            let _e3638_ = bitcast<u32>(_e3566_);
                                            let _e3639_ = (_e3638_ & 2139095040u);
                                            if (_e3639_ != 2139095040u) {
                                                if (_e3566_ <= 0f) {
                                                    phi_16383_ = 0f;
                                                } else {
                                                    if (_e3566_ >= 64f) {
                                                        phi_16382_ = 64f;
                                                    } else {
                                                        if (_e3639_ == 2139095040u) {
                                                            phi_16376_ = true;
                                                        } else {
                                                            phi_16376_ = (_e3639_ == 0u);
                                                        }
                                                        let _e3648_ = phi_16376_;
                                                        phi_16382_ = bitcast<f32>(select(_e3638_, 0u, _e3648_));
                                                    }
                                                    let _e3652_ = phi_16382_;
                                                    phi_16383_ = _e3652_;
                                                }
                                                let _e3654_ = phi_16383_;
                                                phi_16390_ = _e3654_;
                                            } else {
                                                phi_16390_ = select(0f, 64f, (_e3638_ == 2139095040u));
                                            }
                                            let _e3656_ = phi_16390_;
                                            let _e3657_ = bitcast<u32>(_e3573_);
                                            let _e3658_ = (_e3657_ & 2139095040u);
                                            if (_e3658_ != 2139095040u) {
                                                if (_e3573_ <= 0f) {
                                                    phi_16422_ = 0f;
                                                } else {
                                                    if (_e3573_ >= 1f) {
                                                        phi_16421_ = 1f;
                                                    } else {
                                                        if (_e3658_ == 2139095040u) {
                                                            phi_16415_ = true;
                                                        } else {
                                                            phi_16415_ = (_e3658_ == 0u);
                                                        }
                                                        let _e3667_ = phi_16415_;
                                                        phi_16421_ = bitcast<f32>(select(_e3657_, 0u, _e3667_));
                                                    }
                                                    let _e3671_ = phi_16421_;
                                                    phi_16422_ = _e3671_;
                                                }
                                                let _e3673_ = phi_16422_;
                                                phi_16429_ = _e3673_;
                                            } else {
                                                phi_16429_ = select(0f, 1f, (_e3657_ == 2139095040u));
                                            }
                                            let _e3675_ = phi_16429_;
                                            let _e3676_ = bitcast<u32>(_e3580_);
                                            let _e3677_ = (_e3676_ & 2139095040u);
                                            if (_e3677_ != 2139095040u) {
                                                if (_e3580_ <= 0f) {
                                                    phi_16461_ = 0f;
                                                } else {
                                                    if (_e3580_ >= 1f) {
                                                        phi_16460_ = 1f;
                                                    } else {
                                                        if (_e3677_ == 2139095040u) {
                                                            phi_16454_ = true;
                                                        } else {
                                                            phi_16454_ = (_e3677_ == 0u);
                                                        }
                                                        let _e3686_ = phi_16454_;
                                                        phi_16460_ = bitcast<f32>(select(_e3676_, 0u, _e3686_));
                                                    }
                                                    let _e3690_ = phi_16460_;
                                                    phi_16461_ = _e3690_;
                                                }
                                                let _e3692_ = phi_16461_;
                                                phi_16468_ = _e3692_;
                                            } else {
                                                phi_16468_ = select(0f, 1f, (_e3676_ == 2139095040u));
                                            }
                                            let _e3694_ = phi_16468_;
                                            switch bitcast<i32>(_e3538_) {
                                                case 0: {
                                                    phi_16539_ = _e412_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16539_ = _e412_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16539_ = _e412_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16539_ = _e412_.w;
                                                    break;
                                                }
                                            }
                                            let _e3701_ = phi_16539_;
                                            switch bitcast<i32>(_e3538_) {
                                                case 0: {
                                                    phi_16550_ = _e419_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16550_ = _e419_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16550_ = _e419_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16550_ = _e419_.w;
                                                    break;
                                                }
                                            }
                                            let _e3708_ = phi_16550_;
                                            let _e3709_ = (16u + _e157_);
                                            let _e3710_ = (_e3709_ < 256u);
                                            if _e3710_ {
                                            } else {
                                                phi_37452_ = true;
                                                break;
                                            }
                                            let _e3712_ = aux_scratch[_e3709_];
                                            let _e3713_ = (20u + _e157_);
                                            let _e3714_ = (_e3713_ < 256u);
                                            if _e3714_ {
                                            } else {
                                                phi_37452_ = true;
                                                break;
                                            }
                                            let _e3716_ = aux_scratch[_e3713_];
                                            phi_16645_ = type_15_(0u, _e1495_);
                                            phi_16648_ = _e3716_;
                                            phi_16650_ = _e3712_;
                                            loop {
                                                let _e3719_ = phi_16645_;
                                                let _e3721_ = phi_16648_;
                                                let _e3723_ = phi_16650_;
                                                local_33_ = _e3723_;
                                                local_34_ = _e3721_;
                                                if (_e3719_.member < _e3719_.member_1_) {
                                                    phi_16646_ = type_15_((_e3719_.member + 1u), _e3719_.member_1_);
                                                    phi_16673_ = type_15_(1u, _e3719_.member);
                                                } else {
                                                    phi_16646_ = _e3719_;
                                                    phi_16673_ = type_15_(0u, u32());
                                                }
                                                let _e3736_ = phi_16646_;
                                                let _e3738_ = phi_16673_;
                                                let _e3742_ = (bitcast<i32>(_e3738_.member) != 0i);
                                                if _e3742_ {
                                                    let _e3744_ = (_e157_ * 4u);
                                                    let _e3746_ = ((_e3738_.member_1_ & 127u) & 127u);
                                                    let _e3747_ = ((_e157_ * 512u) + _e3746_);
                                                    if (_e3747_ < 2048u) {
                                                    } else {
                                                        phi_35287_ = true;
                                                        break;
                                                    }
                                                    let _e3750_ = lane_buf[_e3747_];
                                                    let _e3753_ = (((_e3744_ + 1u) * 128u) + _e3746_);
                                                    let _e3754_ = (_e3753_ < 2048u);
                                                    if _e3754_ {
                                                    } else {
                                                        phi_35287_ = true;
                                                        break;
                                                    }
                                                    let _e3756_ = lane_buf[_e3753_];
                                                    let _e3759_ = (((_e3744_ + 2u) * 128u) + _e3746_);
                                                    if (_e3759_ < 2048u) {
                                                    } else {
                                                        phi_35287_ = true;
                                                        break;
                                                    }
                                                    let _e3762_ = lane_buf[_e3759_];
                                                    let _e3765_ = (((_e3744_ + 3u) * 128u) + _e3746_);
                                                    let _e3766_ = (_e3765_ < 2048u);
                                                    if _e3766_ {
                                                    } else {
                                                        phi_35287_ = true;
                                                        break;
                                                    }
                                                    let _e3768_ = lane_buf[_e3765_];
                                                    let _e3771_ = bitcast<u32>((_e3701_ * _e3756_));
                                                    let _e3772_ = (_e3771_ & 2139095040u);
                                                    if (_e3772_ == 2139095040u) {
                                                        phi_16744_ = 0u;
                                                    } else {
                                                        if (_e3772_ == 0u) {
                                                            phi_16743_ = (_e3771_ & 2147483648u);
                                                        } else {
                                                            phi_16743_ = _e3771_;
                                                        }
                                                        let _e3777_ = phi_16743_;
                                                        phi_16744_ = _e3777_;
                                                    }
                                                    let _e3779_ = phi_16744_;
                                                    let _e3785_ = (_e3656_ <= 0f);
                                                    let _e3787_ = (1f - _e3618_);
                                                    let _e3788_ = (_e3787_ + select((_e3656_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e3779_)) & 2147483647u))), 0f, _e3785_));
                                                    if (_e3788_ != _e3788_) {
                                                        phi_24951_ = true;
                                                    } else {
                                                        phi_24951_ = (0f >= _e3788_);
                                                    }
                                                    let _e3792_ = phi_24951_;
                                                    let _e3793_ = select(_e3788_, 0f, _e3792_);
                                                    if (_e3793_ != _e3793_) {
                                                        phi_25011_ = true;
                                                    } else {
                                                        phi_25011_ = (1f <= _e3793_);
                                                    }
                                                    let _e3797_ = phi_25011_;
                                                    let _e3798_ = select(_e3793_, 1f, _e3797_);
                                                    let _e3799_ = (_e3599_ * _e3723_);
                                                    let _e3800_ = (1f - _e3675_);
                                                    let _e3801_ = (_e3637_ * _e3800_);
                                                    let _e3803_ = ((_e3708_ * _e3750_) - (_e3801_ * _e3799_));
                                                    if (_e3803_ != _e3803_) {
                                                        phi_25071_ = true;
                                                    } else {
                                                        phi_25071_ = (-1f >= _e3803_);
                                                    }
                                                    let _e3807_ = phi_25071_;
                                                    let _e3808_ = select(_e3803_, -1f, _e3807_);
                                                    if (_e3808_ != _e3808_) {
                                                        phi_25131_ = true;
                                                    } else {
                                                        phi_25131_ = (1f <= _e3808_);
                                                    }
                                                    let _e3812_ = phi_25131_;
                                                    let _e3813_ = select(_e3808_, 1f, _e3812_);
                                                    let _e3820_ = bitcast<f32>((bitcast<u32>(_e3803_) & 2147483647u));
                                                    if (_e3820_ != _e3820_) {
                                                        phi_25191_ = true;
                                                    } else {
                                                        phi_25191_ = (1000000000000000f <= _e3820_);
                                                    }
                                                    let _e3824_ = phi_25191_;
                                                    let _e3825_ = select(_e3820_, 1000000000000000f, _e3824_);
                                                    let _e3826_ = (_e3825_ * _e3825_);
                                                    let _e3827_ = (_e3826_ + 0.0000010000001f);
                                                    if (_e3827_ != _e3827_) {
                                                        phi_25251_ = true;
                                                    } else {
                                                        phi_25251_ = (0.000000000001f >= _e3827_);
                                                    }
                                                    let _e3831_ = phi_25251_;
                                                    let _e3832_ = select(_e3827_, 0.000000000001f, _e3831_);
                                                    let _e3833_ = (_e3832_ * 0.5f);
                                                    let _e3838_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e3832_) >> bitcast<u32>(1i))));
                                                    let _e3842_ = (_e3838_ * (1.5f - (_e3833_ * (_e3838_ * _e3838_))));
                                                    let _e3846_ = (_e3842_ * (1.5f - (_e3833_ * (_e3842_ * _e3842_))));
                                                    let _e3851_ = (_e3827_ * (_e3846_ * (1.5f - (_e3833_ * (_e3846_ * _e3846_)))));
                                                    let _e3852_ = (_e3851_ + 0.001f);
                                                    if (_e3852_ != _e3852_) {
                                                        phi_25311_ = true;
                                                    } else {
                                                        phi_25311_ = (0.000001f >= _e3852_);
                                                    }
                                                    let _e3856_ = phi_25311_;
                                                    let _e3857_ = select(_e3852_, 0.000001f, _e3856_);
                                                    let _e3860_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3857_)));
                                                    let _e3864_ = (_e3860_ + (_e3860_ * (1f - (_e3857_ * _e3860_))));
                                                    let _e3868_ = (_e3864_ + (_e3864_ * (1f - (_e3857_ * _e3864_))));
                                                    let _e3872_ = (_e3868_ + (_e3868_ * (1f - (_e3857_ * _e3868_))));
                                                    let _e3873_ = (_e3851_ + _e3825_);
                                                    if (_e3873_ != _e3873_) {
                                                        phi_25371_ = true;
                                                    } else {
                                                        phi_25371_ = (0.000001f >= _e3873_);
                                                    }
                                                    let _e3877_ = phi_25371_;
                                                    let _e3878_ = select(_e3873_, 0.000001f, _e3877_);
                                                    let _e3881_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3878_)));
                                                    let _e3885_ = (_e3881_ + (_e3881_ * (1f - (_e3878_ * _e3881_))));
                                                    let _e3889_ = (_e3885_ + (_e3885_ * (1f - (_e3878_ * _e3885_))));
                                                    let _e3899_ = (_e3820_ <= 1000000000000000f);
                                                    let _e3909_ = select(select(select(_e3820_, (0.5f * (_e3825_ + (_e3826_ * _e3872_))), _e3899_), select(-0.0005f, -(((0.0005f * (_e3825_ * (_e3889_ + (_e3889_ * (1f - (_e3878_ * _e3889_)))))) * (1f + (_e3825_ * _e3872_)))), _e3899_), (_e3803_ <= 0f)), 0f, (_e3820_ <= 0f));
                                                    let _e3913_ = (_e3675_ <= 0f);
                                                    let _e3920_ = bitcast<u32>((((1f - (_e3675_ * _e3798_)) * _e3799_) + (_e3798_ * select(((_e3800_ * _e3909_) + (_e3675_ * (_e3813_ * (1f - (0.33333334f * (_e3813_ * _e3813_)))))), _e3909_, _e3913_))));
                                                    let _e3921_ = (_e3920_ & 2139095040u);
                                                    if (_e3921_ == 2139095040u) {
                                                        phi_17486_ = 0u;
                                                    } else {
                                                        if (_e3921_ == 0u) {
                                                            phi_17485_ = (_e3920_ & 2147483648u);
                                                        } else {
                                                            phi_17485_ = _e3920_;
                                                        }
                                                        let _e3926_ = phi_17485_;
                                                        phi_17486_ = _e3926_;
                                                    }
                                                    let _e3928_ = phi_17486_;
                                                    let _e3929_ = bitcast<f32>(_e3928_);
                                                    let _e3932_ = bitcast<u32>((_e3701_ * _e3768_));
                                                    let _e3933_ = (_e3932_ & 2139095040u);
                                                    if (_e3933_ == 2139095040u) {
                                                        phi_17548_ = 0u;
                                                    } else {
                                                        if (_e3933_ == 0u) {
                                                            phi_17547_ = (_e3932_ & 2147483648u);
                                                        } else {
                                                            phi_17547_ = _e3932_;
                                                        }
                                                        let _e3938_ = phi_17547_;
                                                        phi_17548_ = _e3938_;
                                                    }
                                                    let _e3940_ = phi_17548_;
                                                    let _e3947_ = (_e3787_ + select((_e3656_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e3940_)) & 2147483647u))), 0f, _e3785_));
                                                    if (_e3947_ != _e3947_) {
                                                        phi_25551_ = true;
                                                    } else {
                                                        phi_25551_ = (0f >= _e3947_);
                                                    }
                                                    let _e3951_ = phi_25551_;
                                                    let _e3952_ = select(_e3947_, 0f, _e3951_);
                                                    if (_e3952_ != _e3952_) {
                                                        phi_25611_ = true;
                                                    } else {
                                                        phi_25611_ = (1f <= _e3952_);
                                                    }
                                                    let _e3956_ = phi_25611_;
                                                    let _e3957_ = select(_e3952_, 1f, _e3956_);
                                                    let _e3958_ = (_e3599_ * _e3721_);
                                                    let _e3960_ = ((_e3708_ * _e3762_) - (_e3801_ * _e3958_));
                                                    if (_e3960_ != _e3960_) {
                                                        phi_25671_ = true;
                                                    } else {
                                                        phi_25671_ = (-1f >= _e3960_);
                                                    }
                                                    let _e3964_ = phi_25671_;
                                                    let _e3965_ = select(_e3960_, -1f, _e3964_);
                                                    if (_e3965_ != _e3965_) {
                                                        phi_25731_ = true;
                                                    } else {
                                                        phi_25731_ = (1f <= _e3965_);
                                                    }
                                                    let _e3969_ = phi_25731_;
                                                    let _e3970_ = select(_e3965_, 1f, _e3969_);
                                                    let _e3977_ = bitcast<f32>((bitcast<u32>(_e3960_) & 2147483647u));
                                                    if (_e3977_ != _e3977_) {
                                                        phi_25791_ = true;
                                                    } else {
                                                        phi_25791_ = (1000000000000000f <= _e3977_);
                                                    }
                                                    let _e3981_ = phi_25791_;
                                                    let _e3982_ = select(_e3977_, 1000000000000000f, _e3981_);
                                                    let _e3983_ = (_e3982_ * _e3982_);
                                                    let _e3984_ = (_e3983_ + 0.0000010000001f);
                                                    if (_e3984_ != _e3984_) {
                                                        phi_25851_ = true;
                                                    } else {
                                                        phi_25851_ = (0.000000000001f >= _e3984_);
                                                    }
                                                    let _e3988_ = phi_25851_;
                                                    let _e3989_ = select(_e3984_, 0.000000000001f, _e3988_);
                                                    let _e3990_ = (_e3989_ * 0.5f);
                                                    let _e3995_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e3989_) >> bitcast<u32>(1i))));
                                                    let _e3999_ = (_e3995_ * (1.5f - (_e3990_ * (_e3995_ * _e3995_))));
                                                    let _e4003_ = (_e3999_ * (1.5f - (_e3990_ * (_e3999_ * _e3999_))));
                                                    let _e4008_ = (_e3984_ * (_e4003_ * (1.5f - (_e3990_ * (_e4003_ * _e4003_)))));
                                                    let _e4009_ = (_e4008_ + 0.001f);
                                                    if (_e4009_ != _e4009_) {
                                                        phi_25911_ = true;
                                                    } else {
                                                        phi_25911_ = (0.000001f >= _e4009_);
                                                    }
                                                    let _e4013_ = phi_25911_;
                                                    let _e4014_ = select(_e4009_, 0.000001f, _e4013_);
                                                    let _e4017_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4014_)));
                                                    let _e4021_ = (_e4017_ + (_e4017_ * (1f - (_e4014_ * _e4017_))));
                                                    let _e4025_ = (_e4021_ + (_e4021_ * (1f - (_e4014_ * _e4021_))));
                                                    let _e4029_ = (_e4025_ + (_e4025_ * (1f - (_e4014_ * _e4025_))));
                                                    let _e4030_ = (_e4008_ + _e3982_);
                                                    if (_e4030_ != _e4030_) {
                                                        phi_25971_ = true;
                                                    } else {
                                                        phi_25971_ = (0.000001f >= _e4030_);
                                                    }
                                                    let _e4034_ = phi_25971_;
                                                    let _e4035_ = select(_e4030_, 0.000001f, _e4034_);
                                                    let _e4038_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4035_)));
                                                    let _e4042_ = (_e4038_ + (_e4038_ * (1f - (_e4035_ * _e4038_))));
                                                    let _e4046_ = (_e4042_ + (_e4042_ * (1f - (_e4035_ * _e4042_))));
                                                    let _e4056_ = (_e3977_ <= 1000000000000000f);
                                                    let _e4066_ = select(select(select(_e3977_, (0.5f * (_e3982_ + (_e3983_ * _e4029_))), _e4056_), select(-0.0005f, -(((0.0005f * (_e3982_ * (_e4046_ + (_e4046_ * (1f - (_e4035_ * _e4046_)))))) * (1f + (_e3982_ * _e4029_)))), _e4056_), (_e3960_ <= 0f)), 0f, (_e3977_ <= 0f));
                                                    let _e4076_ = bitcast<u32>((((1f - (_e3675_ * _e3957_)) * _e3958_) + (_e3957_ * select(((_e3800_ * _e4066_) + (_e3675_ * (_e3970_ * (1f - (0.33333334f * (_e3970_ * _e3970_)))))), _e4066_, _e3913_))));
                                                    let _e4077_ = (_e4076_ & 2139095040u);
                                                    if (_e4077_ == 2139095040u) {
                                                        phi_18290_ = 0u;
                                                    } else {
                                                        if (_e4077_ == 0u) {
                                                            phi_18289_ = (_e4076_ & 2147483648u);
                                                        } else {
                                                            phi_18289_ = _e4076_;
                                                        }
                                                        let _e4082_ = phi_18289_;
                                                        phi_18290_ = _e4082_;
                                                    }
                                                    let _e4084_ = phi_18290_;
                                                    let _e4085_ = bitcast<f32>(_e4084_);
                                                    let _e4087_ = (0.5f * (_e3929_ + _e4085_));
                                                    if _e3754_ {
                                                    } else {
                                                        phi_35287_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e3753_] = (_e3929_ + (_e3694_ * (_e4087_ - _e3929_)));
                                                    if _e3766_ {
                                                    } else {
                                                        phi_35287_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e3765_] = (_e4085_ + (_e3694_ * (_e4087_ - _e4085_)));
                                                    phi_16649_ = _e4085_;
                                                    phi_16651_ = _e3929_;
                                                } else {
                                                    phi_16649_ = f32();
                                                    phi_16651_ = f32();
                                                }
                                                let _e4095_ = phi_16649_;
                                                let _e4097_ = phi_16651_;
                                                phi_16645_ = _e3736_;
                                                phi_16648_ = _e4095_;
                                                phi_16650_ = _e4097_;
                                                phi_35287_ = _e3536_;
                                                if !(_e3742_) {
                                                    break;
                                                }
                                            }
                                            let _e4100_ = phi_35287_;
                                            phi_37452_ = _e4100_;
                                            if _e4100_ {
                                                break;
                                            }
                                            let _e4102_ = local_33_;
                                            let _e4103_ = bitcast<u32>(_e4102_);
                                            let _e4104_ = (_e4103_ & 2139095040u);
                                            if (_e4104_ == 2139095040u) {
                                                phi_18421_ = true;
                                            } else {
                                                phi_18421_ = (_e4104_ == 0u);
                                            }
                                            let _e4108_ = phi_18421_;
                                            if _e3710_ {
                                            } else {
                                                phi_37452_ = true;
                                                break;
                                            }
                                            aux_scratch[_e3709_] = bitcast<f32>(select(_e4103_, 0u, _e4108_));
                                            let _e4112_ = local_34_;
                                            let _e4113_ = bitcast<u32>(_e4112_);
                                            let _e4114_ = (_e4113_ & 2139095040u);
                                            if (_e4114_ == 2139095040u) {
                                                phi_18444_ = true;
                                            } else {
                                                phi_18444_ = (_e4114_ == 0u);
                                            }
                                            let _e4118_ = phi_18444_;
                                            if _e3714_ {
                                            } else {
                                                phi_37452_ = true;
                                                break;
                                            }
                                            aux_scratch[_e3713_] = bitcast<f32>(select(_e4113_, 0u, _e4118_));
                                            phi_36592_ = _e4100_;
                                        } else {
                                            phi_36592_ = _e3536_;
                                        }
                                        let _e4122_ = phi_36592_;
                                        workgroupBarrier();
                                        phi_36591_ = _e4122_;
                                    }
                                    let _e6290_ = phi_36591_;
                                    local_9_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    local_10_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    local_11_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    local_12_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    let _e6291_ = (_e157_ < 128u);
                                    if _e6291_ {
                                        if _e1497_ {
                                            phi_18466_ = type_15_(0u, 4u);
                                            loop {
                                                let _e6293_ = phi_18466_;
                                                if (_e6293_.member < _e6293_.member_1_) {
                                                    phi_18467_ = type_15_((_e6293_.member + 1u), _e6293_.member_1_);
                                                    phi_18490_ = type_15_(1u, _e6293_.member);
                                                } else {
                                                    phi_18467_ = _e6293_;
                                                    phi_18490_ = type_15_(0u, u32());
                                                }
                                                let _e6306_ = phi_18467_;
                                                let _e6308_ = phi_18490_;
                                                let _e6312_ = (bitcast<i32>(_e6308_.member) != 0i);
                                                if _e6312_ {
                                                    let _e6313_ = (_e6308_.member_1_ & 3u);
                                                    switch bitcast<i32>(_e6313_) {
                                                        case 0: {
                                                            phi_18861_ = _e426_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_18861_ = _e426_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_18861_ = _e426_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_18861_ = _e426_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e6320_ = phi_18861_;
                                                    switch bitcast<i32>(_e6313_) {
                                                        case 0: {
                                                            phi_18872_ = _e433_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_18872_ = _e433_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_18872_ = _e433_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_18872_ = _e433_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e6327_ = phi_18872_;
                                                    switch bitcast<i32>(_e6313_) {
                                                        case 0: {
                                                            phi_18883_ = _e440_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_18883_ = _e440_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_18883_ = _e440_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_18883_ = _e440_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e6334_ = phi_18883_;
                                                    let _e6335_ = (_e6308_.member_1_ * 4u);
                                                    let _e6337_ = (_e158_ & 127u);
                                                    let _e6338_ = ((_e6308_.member_1_ * 512u) + _e6337_);
                                                    if (_e6338_ < 2048u) {
                                                    } else {
                                                        phi_36587_ = true;
                                                        break;
                                                    }
                                                    let _e6341_ = lane_buf[_e6338_];
                                                    let _e6344_ = (((_e6335_ + 1u) * 128u) + _e6337_);
                                                    if (_e6344_ < 2048u) {
                                                    } else {
                                                        phi_36587_ = true;
                                                        break;
                                                    }
                                                    let _e6347_ = lane_buf[_e6344_];
                                                    let _e6350_ = (((_e6335_ + 2u) * 128u) + _e6337_);
                                                    if (_e6350_ < 2048u) {
                                                    } else {
                                                        phi_36587_ = true;
                                                        break;
                                                    }
                                                    let _e6353_ = lane_buf[_e6350_];
                                                    let _e6356_ = (((_e6335_ + 3u) * 128u) + _e6337_);
                                                    if (_e6356_ < 2048u) {
                                                    } else {
                                                        phi_36587_ = true;
                                                        break;
                                                    }
                                                    let _e6359_ = lane_buf[_e6356_];
                                                    let _e6360_ = (_e6308_.member_1_ < 4u);
                                                    if _e6360_ {
                                                    } else {
                                                        phi_36587_ = true;
                                                        break;
                                                    }
                                                    local_9_[_e6308_.member_1_] = _e6347_;
                                                    if _e6360_ {
                                                    } else {
                                                        phi_36587_ = true;
                                                        break;
                                                    }
                                                    local_10_[_e6308_.member_1_] = _e6359_;
                                                    let _e6364_ = (1f - (_e6320_ * _e6347_));
                                                    if (_e6364_ != _e6364_) {
                                                        phi_30591_ = true;
                                                    } else {
                                                        phi_30591_ = (0.05f >= _e6364_);
                                                    }
                                                    let _e6368_ = phi_30591_;
                                                    let _e6371_ = (1f - (_e6320_ * _e6359_));
                                                    if (_e6371_ != _e6371_) {
                                                        phi_30606_ = true;
                                                    } else {
                                                        phi_30606_ = (0.05f >= _e6371_);
                                                    }
                                                    let _e6375_ = phi_30606_;
                                                    let _e6381_ = bitcast<u32>((((_e6341_ * _e6327_) * select(_e6364_, 0.05f, _e6368_)) + (_e6334_ * _e6347_)));
                                                    let _e6382_ = (_e6381_ & 2139095040u);
                                                    if (_e6382_ == 2139095040u) {
                                                        phi_19010_ = true;
                                                    } else {
                                                        phi_19010_ = (_e6382_ == 0u);
                                                    }
                                                    let _e6386_ = phi_19010_;
                                                    let _e6388_ = bitcast<f32>(select(_e6381_, 0u, _e6386_));
                                                    let _e6393_ = bitcast<u32>((((_e6353_ * _e6327_) * select(_e6371_, 0.05f, _e6375_)) + (_e6334_ * _e6359_)));
                                                    let _e6394_ = (_e6393_ & 2139095040u);
                                                    if (_e6394_ == 2139095040u) {
                                                        phi_19030_ = true;
                                                    } else {
                                                        phi_19030_ = (_e6394_ == 0u);
                                                    }
                                                    let _e6398_ = phi_19030_;
                                                    let _e6400_ = bitcast<f32>(select(_e6393_, 0u, _e6398_));
                                                    if _e6360_ {
                                                    } else {
                                                        phi_36587_ = true;
                                                        break;
                                                    }
                                                    local_11_[_e6308_.member_1_] = _e6388_;
                                                    if _e6360_ {
                                                    } else {
                                                        phi_36587_ = true;
                                                        break;
                                                    }
                                                    local_12_[_e6308_.member_1_] = _e6400_;
                                                    let _e6404_ = ((_e6308_.member_1_ * 128u) + _e158_);
                                                    if (_e6404_ < 1536u) {
                                                    } else {
                                                        phi_36587_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e6404_] = _e6388_;
                                                    let _e6409_ = (((4u + _e6308_.member_1_) * 128u) + _e158_);
                                                    if (_e6409_ < 1536u) {
                                                    } else {
                                                        phi_36587_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e6409_] = _e6400_;
                                                }
                                                phi_18466_ = _e6306_;
                                                phi_36587_ = _e6290_;
                                                if !(_e6312_) {
                                                    break;
                                                }
                                            }
                                            let _e6414_ = phi_36587_;
                                            phi_37452_ = _e6414_;
                                            if _e6414_ {
                                                break;
                                            }
                                            phi_36613_ = _e6414_;
                                        } else {
                                            phi_36613_ = _e6290_;
                                        }
                                        let _e6416_ = phi_36613_;
                                        phi_36612_ = _e6416_;
                                    } else {
                                        phi_36612_ = _e6290_;
                                    }
                                    let _e6418_ = phi_36612_;
                                    workgroupBarrier();
                                    if _e6291_ {
                                        if _e1497_ {
                                            local_13_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_14_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            if (_e158_ == 0u) {
                                                phi_19078_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e6460_ = phi_19078_;
                                                    if (_e6460_.member < _e6460_.member_1_) {
                                                        phi_19079_ = type_15_((_e6460_.member + 1u), _e6460_.member_1_);
                                                        phi_19102_ = type_15_(1u, _e6460_.member);
                                                    } else {
                                                        phi_19079_ = _e6460_;
                                                        phi_19102_ = type_15_(0u, u32());
                                                    }
                                                    let _e6473_ = phi_19079_;
                                                    let _e6475_ = phi_19102_;
                                                    let _e6479_ = (bitcast<i32>(_e6475_.member) != 0i);
                                                    if _e6479_ {
                                                        let _e6480_ = (24u + _e6475_.member_1_);
                                                        if (_e6480_ < 256u) {
                                                        } else {
                                                            phi_36614_ = true;
                                                            break;
                                                        }
                                                        let _e6483_ = aux_scratch[_e6480_];
                                                        let _e6484_ = (_e6475_.member_1_ < 4u);
                                                        if _e6484_ {
                                                        } else {
                                                            phi_36614_ = true;
                                                            break;
                                                        }
                                                        local_13_[_e6475_.member_1_] = _e6483_;
                                                        let _e6486_ = (28u + _e6475_.member_1_);
                                                        if (_e6486_ < 256u) {
                                                        } else {
                                                            phi_36614_ = true;
                                                            break;
                                                        }
                                                        let _e6489_ = aux_scratch[_e6486_];
                                                        if _e6484_ {
                                                        } else {
                                                            phi_36614_ = true;
                                                            break;
                                                        }
                                                        local_14_[_e6475_.member_1_] = _e6489_;
                                                    }
                                                    phi_19078_ = _e6473_;
                                                    phi_36614_ = _e6418_;
                                                    if !(_e6479_) {
                                                        break;
                                                    }
                                                }
                                                let _e6493_ = phi_36614_;
                                                phi_37452_ = _e6493_;
                                                if _e6493_ {
                                                    break;
                                                }
                                                phi_37437_ = _e6493_;
                                            } else {
                                                let _e6420_ = (_e158_ - 1u);
                                                phi_19141_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e6422_ = phi_19141_;
                                                    if (_e6422_.member < _e6422_.member_1_) {
                                                        phi_19142_ = type_15_((_e6422_.member + 1u), _e6422_.member_1_);
                                                        phi_19165_ = type_15_(1u, _e6422_.member);
                                                    } else {
                                                        phi_19142_ = _e6422_;
                                                        phi_19165_ = type_15_(0u, u32());
                                                    }
                                                    let _e6435_ = phi_19142_;
                                                    let _e6437_ = phi_19165_;
                                                    let _e6441_ = (bitcast<i32>(_e6437_.member) != 0i);
                                                    if _e6441_ {
                                                        let _e6443_ = ((_e6437_.member_1_ * 128u) + _e6420_);
                                                        if (_e6443_ < 1536u) {
                                                        } else {
                                                            phi_36608_ = true;
                                                            break;
                                                        }
                                                        let _e6446_ = scan_scratch[_e6443_];
                                                        let _e6447_ = (_e6437_.member_1_ < 4u);
                                                        if _e6447_ {
                                                        } else {
                                                            phi_36608_ = true;
                                                            break;
                                                        }
                                                        local_13_[_e6437_.member_1_] = _e6446_;
                                                        let _e6451_ = (((4u + _e6437_.member_1_) * 128u) + _e6420_);
                                                        if (_e6451_ < 1536u) {
                                                        } else {
                                                            phi_36608_ = true;
                                                            break;
                                                        }
                                                        let _e6454_ = scan_scratch[_e6451_];
                                                        if _e6447_ {
                                                        } else {
                                                            phi_36608_ = true;
                                                            break;
                                                        }
                                                        local_14_[_e6437_.member_1_] = _e6454_;
                                                    }
                                                    phi_19141_ = _e6435_;
                                                    phi_36608_ = _e6418_;
                                                    if !(_e6441_) {
                                                        break;
                                                    }
                                                }
                                                let _e6458_ = phi_36608_;
                                                phi_37452_ = _e6458_;
                                                if _e6458_ {
                                                    break;
                                                }
                                                phi_37437_ = _e6458_;
                                            }
                                            let _e6495_ = phi_37437_;
                                            let _e6496_ = local_13_;
                                            let _e6501_ = local_11_;
                                            let _e6506_ = local_14_;
                                            let _e6511_ = local_12_;
                                            let _e6529_ = select(0f, _e6496_[0], (bitcast<f32>((bitcast<u32>(_e6496_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6531_ = select(0f, _e6496_[1], (bitcast<f32>((bitcast<u32>(_e6496_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6533_ = select(0f, _e6496_[2], (bitcast<f32>((bitcast<u32>(_e6496_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6535_ = select(0f, _e6496_[3], (bitcast<f32>((bitcast<u32>(_e6496_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6536_ = (_e6529_ != _e6529_);
                                            if _e6536_ {
                                                phi_30621_ = true;
                                            } else {
                                                phi_30621_ = (-1000000000000000000f >= _e6529_);
                                            }
                                            let _e6539_ = phi_30621_;
                                            let _e6540_ = select(_e6529_, -1000000000000000000f, _e6539_);
                                            let _e6541_ = (_e6531_ != _e6531_);
                                            if _e6541_ {
                                                phi_30636_ = true;
                                            } else {
                                                phi_30636_ = (-1000000000000000000f >= _e6531_);
                                            }
                                            let _e6544_ = phi_30636_;
                                            let _e6545_ = select(_e6531_, -1000000000000000000f, _e6544_);
                                            let _e6546_ = (_e6533_ != _e6533_);
                                            if _e6546_ {
                                                phi_30651_ = true;
                                            } else {
                                                phi_30651_ = (-1000000000000000000f >= _e6533_);
                                            }
                                            let _e6549_ = phi_30651_;
                                            let _e6550_ = select(_e6533_, -1000000000000000000f, _e6549_);
                                            let _e6551_ = (_e6535_ != _e6535_);
                                            if _e6551_ {
                                                phi_30666_ = true;
                                            } else {
                                                phi_30666_ = (-1000000000000000000f >= _e6535_);
                                            }
                                            let _e6554_ = phi_30666_;
                                            let _e6555_ = select(_e6535_, -1000000000000000000f, _e6554_);
                                            if (_e6540_ != _e6540_) {
                                                phi_30681_ = true;
                                            } else {
                                                phi_30681_ = (1000000000000000000f <= _e6540_);
                                            }
                                            let _e6559_ = phi_30681_;
                                            let _e6560_ = select(_e6540_, 1000000000000000000f, _e6559_);
                                            if (_e6545_ != _e6545_) {
                                                phi_30696_ = true;
                                            } else {
                                                phi_30696_ = (1000000000000000000f <= _e6545_);
                                            }
                                            let _e6564_ = phi_30696_;
                                            let _e6565_ = select(_e6545_, 1000000000000000000f, _e6564_);
                                            if (_e6550_ != _e6550_) {
                                                phi_30711_ = true;
                                            } else {
                                                phi_30711_ = (1000000000000000000f <= _e6550_);
                                            }
                                            let _e6569_ = phi_30711_;
                                            let _e6570_ = select(_e6550_, 1000000000000000000f, _e6569_);
                                            if (_e6555_ != _e6555_) {
                                                phi_30726_ = true;
                                            } else {
                                                phi_30726_ = (1000000000000000000f <= _e6555_);
                                            }
                                            let _e6574_ = phi_30726_;
                                            let _e6575_ = select(_e6555_, 1000000000000000000f, _e6574_);
                                            let _e6589_ = select(0f, _e6501_[0], (bitcast<f32>((bitcast<u32>(_e6501_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6591_ = select(0f, _e6501_[1], (bitcast<f32>((bitcast<u32>(_e6501_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6593_ = select(0f, _e6501_[2], (bitcast<f32>((bitcast<u32>(_e6501_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6595_ = select(0f, _e6501_[3], (bitcast<f32>((bitcast<u32>(_e6501_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6596_ = (_e6589_ != _e6589_);
                                            if _e6596_ {
                                                phi_30741_ = true;
                                            } else {
                                                phi_30741_ = (-1000000000000000000f >= _e6589_);
                                            }
                                            let _e6599_ = phi_30741_;
                                            let _e6600_ = select(_e6589_, -1000000000000000000f, _e6599_);
                                            let _e6601_ = (_e6591_ != _e6591_);
                                            if _e6601_ {
                                                phi_30756_ = true;
                                            } else {
                                                phi_30756_ = (-1000000000000000000f >= _e6591_);
                                            }
                                            let _e6604_ = phi_30756_;
                                            let _e6605_ = select(_e6591_, -1000000000000000000f, _e6604_);
                                            let _e6606_ = (_e6593_ != _e6593_);
                                            if _e6606_ {
                                                phi_30771_ = true;
                                            } else {
                                                phi_30771_ = (-1000000000000000000f >= _e6593_);
                                            }
                                            let _e6609_ = phi_30771_;
                                            let _e6610_ = select(_e6593_, -1000000000000000000f, _e6609_);
                                            let _e6611_ = (_e6595_ != _e6595_);
                                            if _e6611_ {
                                                phi_30786_ = true;
                                            } else {
                                                phi_30786_ = (-1000000000000000000f >= _e6595_);
                                            }
                                            let _e6614_ = phi_30786_;
                                            let _e6615_ = select(_e6595_, -1000000000000000000f, _e6614_);
                                            if (_e6600_ != _e6600_) {
                                                phi_30801_ = true;
                                            } else {
                                                phi_30801_ = (1000000000000000000f <= _e6600_);
                                            }
                                            let _e6619_ = phi_30801_;
                                            let _e6620_ = select(_e6600_, 1000000000000000000f, _e6619_);
                                            if (_e6605_ != _e6605_) {
                                                phi_30816_ = true;
                                            } else {
                                                phi_30816_ = (1000000000000000000f <= _e6605_);
                                            }
                                            let _e6624_ = phi_30816_;
                                            let _e6625_ = select(_e6605_, 1000000000000000000f, _e6624_);
                                            if (_e6610_ != _e6610_) {
                                                phi_30831_ = true;
                                            } else {
                                                phi_30831_ = (1000000000000000000f <= _e6610_);
                                            }
                                            let _e6629_ = phi_30831_;
                                            let _e6630_ = select(_e6610_, 1000000000000000000f, _e6629_);
                                            if (_e6615_ != _e6615_) {
                                                phi_30846_ = true;
                                            } else {
                                                phi_30846_ = (1000000000000000000f <= _e6615_);
                                            }
                                            let _e6634_ = phi_30846_;
                                            let _e6635_ = select(_e6615_, 1000000000000000000f, _e6634_);
                                            let _e6636_ = array<f32, 4>(_e6620_, _e6625_, _e6630_, _e6635_);
                                            let _e6637_ = (_e6560_ != _e6560_);
                                            if _e6637_ {
                                                phi_30861_ = true;
                                            } else {
                                                phi_30861_ = (_e6620_ <= _e6560_);
                                            }
                                            let _e6640_ = phi_30861_;
                                            let _e6641_ = select(_e6560_, _e6620_, _e6640_);
                                            let _e6642_ = (_e6565_ != _e6565_);
                                            if _e6642_ {
                                                phi_30876_ = true;
                                            } else {
                                                phi_30876_ = (_e6625_ <= _e6565_);
                                            }
                                            let _e6645_ = phi_30876_;
                                            let _e6646_ = select(_e6565_, _e6625_, _e6645_);
                                            let _e6647_ = (_e6570_ != _e6570_);
                                            if _e6647_ {
                                                phi_30891_ = true;
                                            } else {
                                                phi_30891_ = (_e6630_ <= _e6570_);
                                            }
                                            let _e6650_ = phi_30891_;
                                            let _e6651_ = select(_e6570_, _e6630_, _e6650_);
                                            let _e6652_ = (_e6575_ != _e6575_);
                                            if _e6652_ {
                                                phi_30906_ = true;
                                            } else {
                                                phi_30906_ = (_e6635_ <= _e6575_);
                                            }
                                            let _e6655_ = phi_30906_;
                                            let _e6656_ = select(_e6575_, _e6635_, _e6655_);
                                            if _e6637_ {
                                                phi_30921_ = true;
                                            } else {
                                                phi_30921_ = (_e6620_ >= _e6560_);
                                            }
                                            let _e6659_ = phi_30921_;
                                            let _e6660_ = select(_e6560_, _e6620_, _e6659_);
                                            if _e6642_ {
                                                phi_30936_ = true;
                                            } else {
                                                phi_30936_ = (_e6625_ >= _e6565_);
                                            }
                                            let _e6663_ = phi_30936_;
                                            let _e6664_ = select(_e6565_, _e6625_, _e6663_);
                                            if _e6647_ {
                                                phi_30951_ = true;
                                            } else {
                                                phi_30951_ = (_e6630_ >= _e6570_);
                                            }
                                            let _e6667_ = phi_30951_;
                                            let _e6668_ = select(_e6570_, _e6630_, _e6667_);
                                            if _e6652_ {
                                                phi_30966_ = true;
                                            } else {
                                                phi_30966_ = (_e6635_ >= _e6575_);
                                            }
                                            let _e6671_ = phi_30966_;
                                            let _e6672_ = select(_e6575_, _e6635_, _e6671_);
                                            let _e6673_ = (_e6641_ != _e6641_);
                                            if _e6673_ {
                                                phi_30981_ = true;
                                            } else {
                                                phi_30981_ = (-1f >= _e6641_);
                                            }
                                            let _e6676_ = phi_30981_;
                                            let _e6677_ = select(_e6641_, -1f, _e6676_);
                                            let _e6678_ = (_e6646_ != _e6646_);
                                            if _e6678_ {
                                                phi_30996_ = true;
                                            } else {
                                                phi_30996_ = (-1f >= _e6646_);
                                            }
                                            let _e6681_ = phi_30996_;
                                            let _e6682_ = select(_e6646_, -1f, _e6681_);
                                            let _e6683_ = (_e6651_ != _e6651_);
                                            if _e6683_ {
                                                phi_31011_ = true;
                                            } else {
                                                phi_31011_ = (-1f >= _e6651_);
                                            }
                                            let _e6686_ = phi_31011_;
                                            let _e6687_ = select(_e6651_, -1f, _e6686_);
                                            let _e6688_ = (_e6656_ != _e6656_);
                                            if _e6688_ {
                                                phi_31026_ = true;
                                            } else {
                                                phi_31026_ = (-1f >= _e6656_);
                                            }
                                            let _e6691_ = phi_31026_;
                                            let _e6692_ = select(_e6656_, -1f, _e6691_);
                                            if (_e6677_ != _e6677_) {
                                                phi_31041_ = true;
                                            } else {
                                                phi_31041_ = (1f <= _e6677_);
                                            }
                                            let _e6696_ = phi_31041_;
                                            let _e6697_ = select(_e6677_, 1f, _e6696_);
                                            if (_e6682_ != _e6682_) {
                                                phi_31056_ = true;
                                            } else {
                                                phi_31056_ = (1f <= _e6682_);
                                            }
                                            let _e6701_ = phi_31056_;
                                            let _e6702_ = select(_e6682_, 1f, _e6701_);
                                            if (_e6687_ != _e6687_) {
                                                phi_31071_ = true;
                                            } else {
                                                phi_31071_ = (1f <= _e6687_);
                                            }
                                            let _e6706_ = phi_31071_;
                                            let _e6707_ = select(_e6687_, 1f, _e6706_);
                                            if (_e6692_ != _e6692_) {
                                                phi_31086_ = true;
                                            } else {
                                                phi_31086_ = (1f <= _e6692_);
                                            }
                                            let _e6711_ = phi_31086_;
                                            let _e6712_ = select(_e6692_, 1f, _e6711_);
                                            let _e6713_ = (_e6660_ != _e6660_);
                                            if _e6713_ {
                                                phi_31101_ = true;
                                            } else {
                                                phi_31101_ = (-1f >= _e6660_);
                                            }
                                            let _e6716_ = phi_31101_;
                                            let _e6717_ = select(_e6660_, -1f, _e6716_);
                                            let _e6718_ = (_e6664_ != _e6664_);
                                            if _e6718_ {
                                                phi_31116_ = true;
                                            } else {
                                                phi_31116_ = (-1f >= _e6664_);
                                            }
                                            let _e6721_ = phi_31116_;
                                            let _e6722_ = select(_e6664_, -1f, _e6721_);
                                            let _e6723_ = (_e6668_ != _e6668_);
                                            if _e6723_ {
                                                phi_31131_ = true;
                                            } else {
                                                phi_31131_ = (-1f >= _e6668_);
                                            }
                                            let _e6726_ = phi_31131_;
                                            let _e6727_ = select(_e6668_, -1f, _e6726_);
                                            let _e6728_ = (_e6672_ != _e6672_);
                                            if _e6728_ {
                                                phi_31146_ = true;
                                            } else {
                                                phi_31146_ = (-1f >= _e6672_);
                                            }
                                            let _e6731_ = phi_31146_;
                                            let _e6732_ = select(_e6672_, -1f, _e6731_);
                                            if (_e6717_ != _e6717_) {
                                                phi_31161_ = true;
                                            } else {
                                                phi_31161_ = (1f <= _e6717_);
                                            }
                                            let _e6736_ = phi_31161_;
                                            let _e6737_ = select(_e6717_, 1f, _e6736_);
                                            if (_e6722_ != _e6722_) {
                                                phi_31176_ = true;
                                            } else {
                                                phi_31176_ = (1f <= _e6722_);
                                            }
                                            let _e6741_ = phi_31176_;
                                            let _e6742_ = select(_e6722_, 1f, _e6741_);
                                            if (_e6727_ != _e6727_) {
                                                phi_31191_ = true;
                                            } else {
                                                phi_31191_ = (1f <= _e6727_);
                                            }
                                            let _e6746_ = phi_31191_;
                                            let _e6747_ = select(_e6727_, 1f, _e6746_);
                                            if (_e6732_ != _e6732_) {
                                                phi_31206_ = true;
                                            } else {
                                                phi_31206_ = (1f <= _e6732_);
                                            }
                                            let _e6751_ = phi_31206_;
                                            let _e6752_ = select(_e6732_, 1f, _e6751_);
                                            let _e6757_ = (0.5f * (_e6737_ + _e6697_));
                                            let _e6758_ = (0.5f * (_e6742_ + _e6702_));
                                            let _e6759_ = (0.5f * (_e6747_ + _e6707_));
                                            let _e6760_ = (0.5f * (_e6752_ + _e6712_));
                                            let _e6761_ = (_e6737_ - _e6697_);
                                            let _e6762_ = (_e6742_ - _e6702_);
                                            let _e6763_ = (_e6747_ - _e6707_);
                                            let _e6764_ = (_e6752_ - _e6712_);
                                            let _e6765_ = (0.5f * _e6761_);
                                            let _e6766_ = (0.5f * _e6762_);
                                            let _e6767_ = (0.5f * _e6763_);
                                            let _e6768_ = (0.5f * _e6764_);
                                            let _e6769_ = (_e6757_ * _e6757_);
                                            let _e6770_ = (_e6758_ * _e6758_);
                                            let _e6771_ = (_e6759_ * _e6759_);
                                            let _e6772_ = (_e6760_ * _e6760_);
                                            let _e6773_ = (_e6765_ * _e6765_);
                                            let _e6774_ = (_e6766_ * _e6766_);
                                            let _e6775_ = (_e6767_ * _e6767_);
                                            let _e6776_ = (_e6768_ * _e6768_);
                                            if (_e6761_ != _e6761_) {
                                                phi_31221_ = true;
                                            } else {
                                                phi_31221_ = (0f >= _e6761_);
                                            }
                                            let _e6780_ = phi_31221_;
                                            let _e6781_ = select(_e6761_, 0f, _e6780_);
                                            if (_e6762_ != _e6762_) {
                                                phi_31236_ = true;
                                            } else {
                                                phi_31236_ = (0f >= _e6762_);
                                            }
                                            let _e6785_ = phi_31236_;
                                            let _e6786_ = select(_e6762_, 0f, _e6785_);
                                            if (_e6763_ != _e6763_) {
                                                phi_31251_ = true;
                                            } else {
                                                phi_31251_ = (0f >= _e6763_);
                                            }
                                            let _e6790_ = phi_31251_;
                                            let _e6791_ = select(_e6763_, 0f, _e6790_);
                                            if (_e6764_ != _e6764_) {
                                                phi_31266_ = true;
                                            } else {
                                                phi_31266_ = (0f >= _e6764_);
                                            }
                                            let _e6795_ = phi_31266_;
                                            let _e6796_ = select(_e6764_, 0f, _e6795_);
                                            let _e6797_ = (_e6660_ - _e6641_);
                                            let _e6798_ = (_e6664_ - _e6646_);
                                            let _e6799_ = (_e6668_ - _e6651_);
                                            let _e6800_ = (_e6672_ - _e6656_);
                                            if _e6713_ {
                                                phi_31281_ = true;
                                            } else {
                                                phi_31281_ = (-1f <= _e6660_);
                                            }
                                            let _e6803_ = phi_31281_;
                                            if _e6718_ {
                                                phi_31296_ = true;
                                            } else {
                                                phi_31296_ = (-1f <= _e6664_);
                                            }
                                            let _e6807_ = phi_31296_;
                                            if _e6723_ {
                                                phi_31311_ = true;
                                            } else {
                                                phi_31311_ = (-1f <= _e6668_);
                                            }
                                            let _e6811_ = phi_31311_;
                                            if _e6728_ {
                                                phi_31326_ = true;
                                            } else {
                                                phi_31326_ = (-1f <= _e6672_);
                                            }
                                            let _e6815_ = phi_31326_;
                                            let _e6817_ = (select(_e6660_, -1f, _e6803_) - _e6641_);
                                            let _e6818_ = (select(_e6664_, -1f, _e6807_) - _e6646_);
                                            let _e6819_ = (select(_e6668_, -1f, _e6811_) - _e6651_);
                                            let _e6820_ = (select(_e6672_, -1f, _e6815_) - _e6656_);
                                            if (_e6817_ != _e6817_) {
                                                phi_31341_ = true;
                                            } else {
                                                phi_31341_ = (0f >= _e6817_);
                                            }
                                            let _e6824_ = phi_31341_;
                                            if (_e6818_ != _e6818_) {
                                                phi_31356_ = true;
                                            } else {
                                                phi_31356_ = (0f >= _e6818_);
                                            }
                                            let _e6829_ = phi_31356_;
                                            if (_e6819_ != _e6819_) {
                                                phi_31371_ = true;
                                            } else {
                                                phi_31371_ = (0f >= _e6819_);
                                            }
                                            let _e6834_ = phi_31371_;
                                            if (_e6820_ != _e6820_) {
                                                phi_31386_ = true;
                                            } else {
                                                phi_31386_ = (0f >= _e6820_);
                                            }
                                            let _e6839_ = phi_31386_;
                                            if _e6673_ {
                                                phi_31401_ = true;
                                            } else {
                                                phi_31401_ = (1f >= _e6641_);
                                            }
                                            let _e6843_ = phi_31401_;
                                            if _e6678_ {
                                                phi_31416_ = true;
                                            } else {
                                                phi_31416_ = (1f >= _e6646_);
                                            }
                                            let _e6847_ = phi_31416_;
                                            if _e6683_ {
                                                phi_31431_ = true;
                                            } else {
                                                phi_31431_ = (1f >= _e6651_);
                                            }
                                            let _e6851_ = phi_31431_;
                                            if _e6688_ {
                                                phi_31446_ = true;
                                            } else {
                                                phi_31446_ = (1f >= _e6656_);
                                            }
                                            let _e6855_ = phi_31446_;
                                            let _e6857_ = (_e6660_ - select(_e6641_, 1f, _e6843_));
                                            let _e6858_ = (_e6664_ - select(_e6646_, 1f, _e6847_));
                                            let _e6859_ = (_e6668_ - select(_e6651_, 1f, _e6851_));
                                            let _e6860_ = (_e6672_ - select(_e6656_, 1f, _e6855_));
                                            if (_e6857_ != _e6857_) {
                                                phi_31461_ = true;
                                            } else {
                                                phi_31461_ = (0f >= _e6857_);
                                            }
                                            let _e6864_ = phi_31461_;
                                            if (_e6858_ != _e6858_) {
                                                phi_31476_ = true;
                                            } else {
                                                phi_31476_ = (0f >= _e6858_);
                                            }
                                            let _e6869_ = phi_31476_;
                                            if (_e6859_ != _e6859_) {
                                                phi_31491_ = true;
                                            } else {
                                                phi_31491_ = (0f >= _e6859_);
                                            }
                                            let _e6874_ = phi_31491_;
                                            if (_e6860_ != _e6860_) {
                                                phi_31506_ = true;
                                            } else {
                                                phi_31506_ = (0f >= _e6860_);
                                            }
                                            let _e6879_ = phi_31506_;
                                            let _e6883_ = bitcast<f32>((bitcast<u32>(_e6560_) & 2147483647u));
                                            let _e6886_ = bitcast<f32>((bitcast<u32>(_e6565_) & 2147483647u));
                                            let _e6889_ = bitcast<f32>((bitcast<u32>(_e6570_) & 2147483647u));
                                            let _e6892_ = bitcast<f32>((bitcast<u32>(_e6575_) & 2147483647u));
                                            let _e6895_ = bitcast<f32>((bitcast<u32>(_e6620_) & 2147483647u));
                                            let _e6898_ = bitcast<f32>((bitcast<u32>(_e6625_) & 2147483647u));
                                            let _e6901_ = bitcast<f32>((bitcast<u32>(_e6630_) & 2147483647u));
                                            let _e6904_ = bitcast<f32>((bitcast<u32>(_e6635_) & 2147483647u));
                                            if (_e6883_ != _e6883_) {
                                                phi_31521_ = true;
                                            } else {
                                                phi_31521_ = (_e6895_ >= _e6883_);
                                            }
                                            let _e6908_ = phi_31521_;
                                            let _e6909_ = select(_e6883_, _e6895_, _e6908_);
                                            if (_e6886_ != _e6886_) {
                                                phi_31536_ = true;
                                            } else {
                                                phi_31536_ = (_e6898_ >= _e6886_);
                                            }
                                            let _e6913_ = phi_31536_;
                                            let _e6914_ = select(_e6886_, _e6898_, _e6913_);
                                            if (_e6889_ != _e6889_) {
                                                phi_31551_ = true;
                                            } else {
                                                phi_31551_ = (_e6901_ >= _e6889_);
                                            }
                                            let _e6918_ = phi_31551_;
                                            let _e6919_ = select(_e6889_, _e6901_, _e6918_);
                                            if (_e6892_ != _e6892_) {
                                                phi_31566_ = true;
                                            } else {
                                                phi_31566_ = (_e6904_ >= _e6892_);
                                            }
                                            let _e6923_ = phi_31566_;
                                            let _e6924_ = select(_e6892_, _e6904_, _e6923_);
                                            let _e6941_ = (_e6909_ <= 0f);
                                            let _e6942_ = select((_e6757_ * (1.5f - (0.5f * (_e6769_ + _e6773_)))), 0f, _e6941_);
                                            let _e6943_ = (_e6914_ <= 0f);
                                            let _e6944_ = select((_e6758_ * (1.5f - (0.5f * (_e6770_ + _e6774_)))), 0f, _e6943_);
                                            let _e6945_ = (_e6919_ <= 0f);
                                            let _e6946_ = select((_e6759_ * (1.5f - (0.5f * (_e6771_ + _e6775_)))), 0f, _e6945_);
                                            let _e6947_ = (_e6924_ <= 0f);
                                            let _e6948_ = select((_e6760_ * (1.5f - (0.5f * (_e6772_ + _e6776_)))), 0f, _e6947_);
                                            let _e6990_ = select(((_e6769_ + (0.33333334f * _e6773_)) - (((_e6769_ * _e6769_) + ((2f * _e6769_) * _e6773_)) + (0.2f * (_e6773_ * _e6773_)))), 0f, _e6941_);
                                            let _e6991_ = select(((_e6770_ + (0.33333334f * _e6774_)) - (((_e6770_ * _e6770_) + ((2f * _e6770_) * _e6774_)) + (0.2f * (_e6774_ * _e6774_)))), 0f, _e6943_);
                                            let _e6992_ = select(((_e6771_ + (0.33333334f * _e6775_)) - (((_e6771_ * _e6771_) + ((2f * _e6771_) * _e6775_)) + (0.2f * (_e6775_ * _e6775_)))), 0f, _e6945_);
                                            let _e6993_ = select(((_e6772_ + (0.33333334f * _e6776_)) - (((_e6772_ * _e6772_) + ((2f * _e6772_) * _e6776_)) + (0.2f * (_e6776_ * _e6776_)))), 0f, _e6947_);
                                            if (_e6909_ <= 1f) {
                                                if (_e6914_ <= 1f) {
                                                    let _e6997_ = (_e6919_ <= 1f);
                                                    if _e6997_ {
                                                        phi_19670_ = (_e6924_ <= 1f);
                                                    } else {
                                                        phi_19670_ = bool();
                                                    }
                                                    let _e7000_ = phi_19670_;
                                                    phi_19672_ = _e7000_;
                                                    phi_19673_ = select(true, false, _e6997_);
                                                } else {
                                                    phi_19672_ = bool();
                                                    phi_19673_ = true;
                                                }
                                                let _e7003_ = phi_19672_;
                                                let _e7005_ = phi_19673_;
                                                phi_19674_ = _e7003_;
                                                phi_19675_ = _e7005_;
                                            } else {
                                                phi_19674_ = bool();
                                                phi_19675_ = true;
                                            }
                                            let _e7007_ = phi_19674_;
                                            let _e7009_ = phi_19675_;
                                            if select(_e7007_, false, _e7009_) {
                                                phi_20053_ = type_37_(type_30_(array<f32, 4>(_e6942_, _e6944_, _e6946_, _e6948_)), type_30_(array<f32, 4>(_e6990_, _e6991_, _e6992_, _e6993_)), type_30_(_e6636_));
                                            } else {
                                                if (_e6797_ != _e6797_) {
                                                    phi_31581_ = true;
                                                } else {
                                                    phi_31581_ = (0.000001f >= _e6797_);
                                                }
                                                let _e7014_ = phi_31581_;
                                                let _e7015_ = select(_e6797_, 0.000001f, _e7014_);
                                                if (_e6798_ != _e6798_) {
                                                    phi_31596_ = true;
                                                } else {
                                                    phi_31596_ = (0.000001f >= _e6798_);
                                                }
                                                let _e7019_ = phi_31596_;
                                                let _e7020_ = select(_e6798_, 0.000001f, _e7019_);
                                                if (_e6799_ != _e6799_) {
                                                    phi_31611_ = true;
                                                } else {
                                                    phi_31611_ = (0.000001f >= _e6799_);
                                                }
                                                let _e7024_ = phi_31611_;
                                                let _e7025_ = select(_e6799_, 0.000001f, _e7024_);
                                                if (_e6800_ != _e6800_) {
                                                    phi_31626_ = true;
                                                } else {
                                                    phi_31626_ = (0.000001f >= _e6800_);
                                                }
                                                let _e7029_ = phi_31626_;
                                                let _e7030_ = select(_e6800_, 0.000001f, _e7029_);
                                                let _e7033_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7015_)));
                                                let _e7036_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7020_)));
                                                let _e7039_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7025_)));
                                                let _e7042_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7030_)));
                                                let _e7052_ = (_e7033_ + (_e7033_ * (1f - (_e7015_ * _e7033_))));
                                                let _e7054_ = (_e7036_ + (_e7036_ * (1f - (_e7020_ * _e7036_))));
                                                let _e7056_ = (_e7039_ + (_e7039_ * (1f - (_e7025_ * _e7039_))));
                                                let _e7058_ = (_e7042_ + (_e7042_ * (1f - (_e7030_ * _e7042_))));
                                                let _e7068_ = (_e7052_ + (_e7052_ * (1f - (_e7015_ * _e7052_))));
                                                let _e7070_ = (_e7054_ + (_e7054_ * (1f - (_e7020_ * _e7054_))));
                                                let _e7072_ = (_e7056_ + (_e7056_ * (1f - (_e7025_ * _e7056_))));
                                                let _e7074_ = (_e7058_ + (_e7058_ * (1f - (_e7030_ * _e7058_))));
                                                let _e7127_ = (select((select(0f, 1f, (-1f <= _e6641_)) * select(0f, 1f, (_e6660_ <= 1f))), 1f, (_e6797_ <= 0.000001f)) <= 0f);
                                                let _e7129_ = (select((select(0f, 1f, (-1f <= _e6646_)) * select(0f, 1f, (_e6664_ <= 1f))), 1f, (_e6798_ <= 0.000001f)) <= 0f);
                                                let _e7131_ = (select((select(0f, 1f, (-1f <= _e6651_)) * select(0f, 1f, (_e6668_ <= 1f))), 1f, (_e6799_ <= 0.000001f)) <= 0f);
                                                let _e7133_ = (select((select(0f, 1f, (-1f <= _e6656_)) * select(0f, 1f, (_e6672_ <= 1f))), 1f, (_e6800_ <= 0.000001f)) <= 0f);
                                                let _e7139_ = (1f <= _e6641_);
                                                let _e7141_ = (1f <= _e6646_);
                                                let _e7143_ = (1f <= _e6651_);
                                                let _e7145_ = (1f <= _e6656_);
                                                let _e7147_ = (_e6660_ <= -1f);
                                                let _e7149_ = (_e6664_ <= -1f);
                                                let _e7151_ = (_e6668_ <= -1f);
                                                let _e7153_ = (_e6672_ <= -1f);
                                                phi_20053_ = type_37_(type_30_(array<f32, 4>(select(select(select(_e6942_, (((select(_e6857_, 0f, _e6864_) - select(_e6817_, 0f, _e6824_)) + (_e6781_ * _e6942_)) * _e7068_), _e7127_), 1f, _e7139_), -1f, _e7147_), select(select(select(_e6944_, (((select(_e6858_, 0f, _e6869_) - select(_e6818_, 0f, _e6829_)) + (_e6786_ * _e6944_)) * _e7070_), _e7129_), 1f, _e7141_), -1f, _e7149_), select(select(select(_e6946_, (((select(_e6859_, 0f, _e6874_) - select(_e6819_, 0f, _e6834_)) + (_e6791_ * _e6946_)) * _e7072_), _e7131_), 1f, _e7143_), -1f, _e7151_), select(select(select(_e6948_, (((select(_e6860_, 0f, _e6879_) - select(_e6820_, 0f, _e6839_)) + (_e6796_ * _e6948_)) * _e7074_), _e7133_), 1f, _e7145_), -1f, _e7153_))), type_30_(array<f32, 4>(select(select(select(_e6990_, ((_e6781_ * _e6990_) * _e7068_), _e7127_), 0f, _e7139_), 0f, _e7147_), select(select(select(_e6991_, ((_e6786_ * _e6991_) * _e7070_), _e7129_), 0f, _e7141_), 0f, _e7149_), select(select(select(_e6992_, ((_e6791_ * _e6992_) * _e7072_), _e7131_), 0f, _e7143_), 0f, _e7151_), select(select(select(_e6993_, ((_e6796_ * _e6993_) * _e7074_), _e7133_), 0f, _e7145_), 0f, _e7153_))), type_30_(_e6636_));
                                            }
                                            let _e7174_ = phi_20053_;
                                            let _e7189_ = select(0f, _e6506_[0], (bitcast<f32>((bitcast<u32>(_e6506_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7191_ = select(0f, _e6506_[1], (bitcast<f32>((bitcast<u32>(_e6506_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7193_ = select(0f, _e6506_[2], (bitcast<f32>((bitcast<u32>(_e6506_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7195_ = select(0f, _e6506_[3], (bitcast<f32>((bitcast<u32>(_e6506_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7196_ = (_e7189_ != _e7189_);
                                            if _e7196_ {
                                                phi_31641_ = true;
                                            } else {
                                                phi_31641_ = (-1000000000000000000f >= _e7189_);
                                            }
                                            let _e7199_ = phi_31641_;
                                            let _e7200_ = select(_e7189_, -1000000000000000000f, _e7199_);
                                            let _e7201_ = (_e7191_ != _e7191_);
                                            if _e7201_ {
                                                phi_31656_ = true;
                                            } else {
                                                phi_31656_ = (-1000000000000000000f >= _e7191_);
                                            }
                                            let _e7204_ = phi_31656_;
                                            let _e7205_ = select(_e7191_, -1000000000000000000f, _e7204_);
                                            let _e7206_ = (_e7193_ != _e7193_);
                                            if _e7206_ {
                                                phi_31671_ = true;
                                            } else {
                                                phi_31671_ = (-1000000000000000000f >= _e7193_);
                                            }
                                            let _e7209_ = phi_31671_;
                                            let _e7210_ = select(_e7193_, -1000000000000000000f, _e7209_);
                                            let _e7211_ = (_e7195_ != _e7195_);
                                            if _e7211_ {
                                                phi_31686_ = true;
                                            } else {
                                                phi_31686_ = (-1000000000000000000f >= _e7195_);
                                            }
                                            let _e7214_ = phi_31686_;
                                            let _e7215_ = select(_e7195_, -1000000000000000000f, _e7214_);
                                            if (_e7200_ != _e7200_) {
                                                phi_31701_ = true;
                                            } else {
                                                phi_31701_ = (1000000000000000000f <= _e7200_);
                                            }
                                            let _e7219_ = phi_31701_;
                                            let _e7220_ = select(_e7200_, 1000000000000000000f, _e7219_);
                                            if (_e7205_ != _e7205_) {
                                                phi_31716_ = true;
                                            } else {
                                                phi_31716_ = (1000000000000000000f <= _e7205_);
                                            }
                                            let _e7224_ = phi_31716_;
                                            let _e7225_ = select(_e7205_, 1000000000000000000f, _e7224_);
                                            if (_e7210_ != _e7210_) {
                                                phi_31731_ = true;
                                            } else {
                                                phi_31731_ = (1000000000000000000f <= _e7210_);
                                            }
                                            let _e7229_ = phi_31731_;
                                            let _e7230_ = select(_e7210_, 1000000000000000000f, _e7229_);
                                            if (_e7215_ != _e7215_) {
                                                phi_31746_ = true;
                                            } else {
                                                phi_31746_ = (1000000000000000000f <= _e7215_);
                                            }
                                            let _e7234_ = phi_31746_;
                                            let _e7235_ = select(_e7215_, 1000000000000000000f, _e7234_);
                                            let _e7249_ = select(0f, _e6511_[0], (bitcast<f32>((bitcast<u32>(_e6511_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7251_ = select(0f, _e6511_[1], (bitcast<f32>((bitcast<u32>(_e6511_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7253_ = select(0f, _e6511_[2], (bitcast<f32>((bitcast<u32>(_e6511_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7255_ = select(0f, _e6511_[3], (bitcast<f32>((bitcast<u32>(_e6511_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7256_ = (_e7249_ != _e7249_);
                                            if _e7256_ {
                                                phi_31761_ = true;
                                            } else {
                                                phi_31761_ = (-1000000000000000000f >= _e7249_);
                                            }
                                            let _e7259_ = phi_31761_;
                                            let _e7260_ = select(_e7249_, -1000000000000000000f, _e7259_);
                                            let _e7261_ = (_e7251_ != _e7251_);
                                            if _e7261_ {
                                                phi_31776_ = true;
                                            } else {
                                                phi_31776_ = (-1000000000000000000f >= _e7251_);
                                            }
                                            let _e7264_ = phi_31776_;
                                            let _e7265_ = select(_e7251_, -1000000000000000000f, _e7264_);
                                            let _e7266_ = (_e7253_ != _e7253_);
                                            if _e7266_ {
                                                phi_31791_ = true;
                                            } else {
                                                phi_31791_ = (-1000000000000000000f >= _e7253_);
                                            }
                                            let _e7269_ = phi_31791_;
                                            let _e7270_ = select(_e7253_, -1000000000000000000f, _e7269_);
                                            let _e7271_ = (_e7255_ != _e7255_);
                                            if _e7271_ {
                                                phi_31806_ = true;
                                            } else {
                                                phi_31806_ = (-1000000000000000000f >= _e7255_);
                                            }
                                            let _e7274_ = phi_31806_;
                                            let _e7275_ = select(_e7255_, -1000000000000000000f, _e7274_);
                                            if (_e7260_ != _e7260_) {
                                                phi_31821_ = true;
                                            } else {
                                                phi_31821_ = (1000000000000000000f <= _e7260_);
                                            }
                                            let _e7279_ = phi_31821_;
                                            let _e7280_ = select(_e7260_, 1000000000000000000f, _e7279_);
                                            if (_e7265_ != _e7265_) {
                                                phi_31836_ = true;
                                            } else {
                                                phi_31836_ = (1000000000000000000f <= _e7265_);
                                            }
                                            let _e7284_ = phi_31836_;
                                            let _e7285_ = select(_e7265_, 1000000000000000000f, _e7284_);
                                            if (_e7270_ != _e7270_) {
                                                phi_31851_ = true;
                                            } else {
                                                phi_31851_ = (1000000000000000000f <= _e7270_);
                                            }
                                            let _e7289_ = phi_31851_;
                                            let _e7290_ = select(_e7270_, 1000000000000000000f, _e7289_);
                                            if (_e7275_ != _e7275_) {
                                                phi_31866_ = true;
                                            } else {
                                                phi_31866_ = (1000000000000000000f <= _e7275_);
                                            }
                                            let _e7294_ = phi_31866_;
                                            let _e7295_ = select(_e7275_, 1000000000000000000f, _e7294_);
                                            let _e7296_ = array<f32, 4>(_e7280_, _e7285_, _e7290_, _e7295_);
                                            let _e7297_ = (_e7220_ != _e7220_);
                                            if _e7297_ {
                                                phi_31881_ = true;
                                            } else {
                                                phi_31881_ = (_e7280_ <= _e7220_);
                                            }
                                            let _e7300_ = phi_31881_;
                                            let _e7301_ = select(_e7220_, _e7280_, _e7300_);
                                            let _e7302_ = (_e7225_ != _e7225_);
                                            if _e7302_ {
                                                phi_31896_ = true;
                                            } else {
                                                phi_31896_ = (_e7285_ <= _e7225_);
                                            }
                                            let _e7305_ = phi_31896_;
                                            let _e7306_ = select(_e7225_, _e7285_, _e7305_);
                                            let _e7307_ = (_e7230_ != _e7230_);
                                            if _e7307_ {
                                                phi_31911_ = true;
                                            } else {
                                                phi_31911_ = (_e7290_ <= _e7230_);
                                            }
                                            let _e7310_ = phi_31911_;
                                            let _e7311_ = select(_e7230_, _e7290_, _e7310_);
                                            let _e7312_ = (_e7235_ != _e7235_);
                                            if _e7312_ {
                                                phi_31926_ = true;
                                            } else {
                                                phi_31926_ = (_e7295_ <= _e7235_);
                                            }
                                            let _e7315_ = phi_31926_;
                                            let _e7316_ = select(_e7235_, _e7295_, _e7315_);
                                            if _e7297_ {
                                                phi_31941_ = true;
                                            } else {
                                                phi_31941_ = (_e7280_ >= _e7220_);
                                            }
                                            let _e7319_ = phi_31941_;
                                            let _e7320_ = select(_e7220_, _e7280_, _e7319_);
                                            if _e7302_ {
                                                phi_31956_ = true;
                                            } else {
                                                phi_31956_ = (_e7285_ >= _e7225_);
                                            }
                                            let _e7323_ = phi_31956_;
                                            let _e7324_ = select(_e7225_, _e7285_, _e7323_);
                                            if _e7307_ {
                                                phi_31971_ = true;
                                            } else {
                                                phi_31971_ = (_e7290_ >= _e7230_);
                                            }
                                            let _e7327_ = phi_31971_;
                                            let _e7328_ = select(_e7230_, _e7290_, _e7327_);
                                            if _e7312_ {
                                                phi_31986_ = true;
                                            } else {
                                                phi_31986_ = (_e7295_ >= _e7235_);
                                            }
                                            let _e7331_ = phi_31986_;
                                            let _e7332_ = select(_e7235_, _e7295_, _e7331_);
                                            let _e7333_ = (_e7301_ != _e7301_);
                                            if _e7333_ {
                                                phi_32001_ = true;
                                            } else {
                                                phi_32001_ = (-1f >= _e7301_);
                                            }
                                            let _e7336_ = phi_32001_;
                                            let _e7337_ = select(_e7301_, -1f, _e7336_);
                                            let _e7338_ = (_e7306_ != _e7306_);
                                            if _e7338_ {
                                                phi_32016_ = true;
                                            } else {
                                                phi_32016_ = (-1f >= _e7306_);
                                            }
                                            let _e7341_ = phi_32016_;
                                            let _e7342_ = select(_e7306_, -1f, _e7341_);
                                            let _e7343_ = (_e7311_ != _e7311_);
                                            if _e7343_ {
                                                phi_32031_ = true;
                                            } else {
                                                phi_32031_ = (-1f >= _e7311_);
                                            }
                                            let _e7346_ = phi_32031_;
                                            let _e7347_ = select(_e7311_, -1f, _e7346_);
                                            let _e7348_ = (_e7316_ != _e7316_);
                                            if _e7348_ {
                                                phi_32046_ = true;
                                            } else {
                                                phi_32046_ = (-1f >= _e7316_);
                                            }
                                            let _e7351_ = phi_32046_;
                                            let _e7352_ = select(_e7316_, -1f, _e7351_);
                                            if (_e7337_ != _e7337_) {
                                                phi_32061_ = true;
                                            } else {
                                                phi_32061_ = (1f <= _e7337_);
                                            }
                                            let _e7356_ = phi_32061_;
                                            let _e7357_ = select(_e7337_, 1f, _e7356_);
                                            if (_e7342_ != _e7342_) {
                                                phi_32076_ = true;
                                            } else {
                                                phi_32076_ = (1f <= _e7342_);
                                            }
                                            let _e7361_ = phi_32076_;
                                            let _e7362_ = select(_e7342_, 1f, _e7361_);
                                            if (_e7347_ != _e7347_) {
                                                phi_32091_ = true;
                                            } else {
                                                phi_32091_ = (1f <= _e7347_);
                                            }
                                            let _e7366_ = phi_32091_;
                                            let _e7367_ = select(_e7347_, 1f, _e7366_);
                                            if (_e7352_ != _e7352_) {
                                                phi_32106_ = true;
                                            } else {
                                                phi_32106_ = (1f <= _e7352_);
                                            }
                                            let _e7371_ = phi_32106_;
                                            let _e7372_ = select(_e7352_, 1f, _e7371_);
                                            let _e7373_ = (_e7320_ != _e7320_);
                                            if _e7373_ {
                                                phi_32121_ = true;
                                            } else {
                                                phi_32121_ = (-1f >= _e7320_);
                                            }
                                            let _e7376_ = phi_32121_;
                                            let _e7377_ = select(_e7320_, -1f, _e7376_);
                                            let _e7378_ = (_e7324_ != _e7324_);
                                            if _e7378_ {
                                                phi_32136_ = true;
                                            } else {
                                                phi_32136_ = (-1f >= _e7324_);
                                            }
                                            let _e7381_ = phi_32136_;
                                            let _e7382_ = select(_e7324_, -1f, _e7381_);
                                            let _e7383_ = (_e7328_ != _e7328_);
                                            if _e7383_ {
                                                phi_32151_ = true;
                                            } else {
                                                phi_32151_ = (-1f >= _e7328_);
                                            }
                                            let _e7386_ = phi_32151_;
                                            let _e7387_ = select(_e7328_, -1f, _e7386_);
                                            let _e7388_ = (_e7332_ != _e7332_);
                                            if _e7388_ {
                                                phi_32166_ = true;
                                            } else {
                                                phi_32166_ = (-1f >= _e7332_);
                                            }
                                            let _e7391_ = phi_32166_;
                                            let _e7392_ = select(_e7332_, -1f, _e7391_);
                                            if (_e7377_ != _e7377_) {
                                                phi_32181_ = true;
                                            } else {
                                                phi_32181_ = (1f <= _e7377_);
                                            }
                                            let _e7396_ = phi_32181_;
                                            let _e7397_ = select(_e7377_, 1f, _e7396_);
                                            if (_e7382_ != _e7382_) {
                                                phi_32196_ = true;
                                            } else {
                                                phi_32196_ = (1f <= _e7382_);
                                            }
                                            let _e7401_ = phi_32196_;
                                            let _e7402_ = select(_e7382_, 1f, _e7401_);
                                            if (_e7387_ != _e7387_) {
                                                phi_32211_ = true;
                                            } else {
                                                phi_32211_ = (1f <= _e7387_);
                                            }
                                            let _e7406_ = phi_32211_;
                                            let _e7407_ = select(_e7387_, 1f, _e7406_);
                                            if (_e7392_ != _e7392_) {
                                                phi_32226_ = true;
                                            } else {
                                                phi_32226_ = (1f <= _e7392_);
                                            }
                                            let _e7411_ = phi_32226_;
                                            let _e7412_ = select(_e7392_, 1f, _e7411_);
                                            let _e7417_ = (0.5f * (_e7397_ + _e7357_));
                                            let _e7418_ = (0.5f * (_e7402_ + _e7362_));
                                            let _e7419_ = (0.5f * (_e7407_ + _e7367_));
                                            let _e7420_ = (0.5f * (_e7412_ + _e7372_));
                                            let _e7421_ = (_e7397_ - _e7357_);
                                            let _e7422_ = (_e7402_ - _e7362_);
                                            let _e7423_ = (_e7407_ - _e7367_);
                                            let _e7424_ = (_e7412_ - _e7372_);
                                            let _e7425_ = (0.5f * _e7421_);
                                            let _e7426_ = (0.5f * _e7422_);
                                            let _e7427_ = (0.5f * _e7423_);
                                            let _e7428_ = (0.5f * _e7424_);
                                            let _e7429_ = (_e7417_ * _e7417_);
                                            let _e7430_ = (_e7418_ * _e7418_);
                                            let _e7431_ = (_e7419_ * _e7419_);
                                            let _e7432_ = (_e7420_ * _e7420_);
                                            let _e7433_ = (_e7425_ * _e7425_);
                                            let _e7434_ = (_e7426_ * _e7426_);
                                            let _e7435_ = (_e7427_ * _e7427_);
                                            let _e7436_ = (_e7428_ * _e7428_);
                                            if (_e7421_ != _e7421_) {
                                                phi_32241_ = true;
                                            } else {
                                                phi_32241_ = (0f >= _e7421_);
                                            }
                                            let _e7440_ = phi_32241_;
                                            let _e7441_ = select(_e7421_, 0f, _e7440_);
                                            if (_e7422_ != _e7422_) {
                                                phi_32256_ = true;
                                            } else {
                                                phi_32256_ = (0f >= _e7422_);
                                            }
                                            let _e7445_ = phi_32256_;
                                            let _e7446_ = select(_e7422_, 0f, _e7445_);
                                            if (_e7423_ != _e7423_) {
                                                phi_32271_ = true;
                                            } else {
                                                phi_32271_ = (0f >= _e7423_);
                                            }
                                            let _e7450_ = phi_32271_;
                                            let _e7451_ = select(_e7423_, 0f, _e7450_);
                                            if (_e7424_ != _e7424_) {
                                                phi_32286_ = true;
                                            } else {
                                                phi_32286_ = (0f >= _e7424_);
                                            }
                                            let _e7455_ = phi_32286_;
                                            let _e7456_ = select(_e7424_, 0f, _e7455_);
                                            let _e7457_ = (_e7320_ - _e7301_);
                                            let _e7458_ = (_e7324_ - _e7306_);
                                            let _e7459_ = (_e7328_ - _e7311_);
                                            let _e7460_ = (_e7332_ - _e7316_);
                                            if _e7373_ {
                                                phi_32301_ = true;
                                            } else {
                                                phi_32301_ = (-1f <= _e7320_);
                                            }
                                            let _e7463_ = phi_32301_;
                                            if _e7378_ {
                                                phi_32316_ = true;
                                            } else {
                                                phi_32316_ = (-1f <= _e7324_);
                                            }
                                            let _e7467_ = phi_32316_;
                                            if _e7383_ {
                                                phi_32331_ = true;
                                            } else {
                                                phi_32331_ = (-1f <= _e7328_);
                                            }
                                            let _e7471_ = phi_32331_;
                                            if _e7388_ {
                                                phi_32346_ = true;
                                            } else {
                                                phi_32346_ = (-1f <= _e7332_);
                                            }
                                            let _e7475_ = phi_32346_;
                                            let _e7477_ = (select(_e7320_, -1f, _e7463_) - _e7301_);
                                            let _e7478_ = (select(_e7324_, -1f, _e7467_) - _e7306_);
                                            let _e7479_ = (select(_e7328_, -1f, _e7471_) - _e7311_);
                                            let _e7480_ = (select(_e7332_, -1f, _e7475_) - _e7316_);
                                            if (_e7477_ != _e7477_) {
                                                phi_32361_ = true;
                                            } else {
                                                phi_32361_ = (0f >= _e7477_);
                                            }
                                            let _e7484_ = phi_32361_;
                                            if (_e7478_ != _e7478_) {
                                                phi_32376_ = true;
                                            } else {
                                                phi_32376_ = (0f >= _e7478_);
                                            }
                                            let _e7489_ = phi_32376_;
                                            if (_e7479_ != _e7479_) {
                                                phi_32391_ = true;
                                            } else {
                                                phi_32391_ = (0f >= _e7479_);
                                            }
                                            let _e7494_ = phi_32391_;
                                            if (_e7480_ != _e7480_) {
                                                phi_32406_ = true;
                                            } else {
                                                phi_32406_ = (0f >= _e7480_);
                                            }
                                            let _e7499_ = phi_32406_;
                                            if _e7333_ {
                                                phi_32421_ = true;
                                            } else {
                                                phi_32421_ = (1f >= _e7301_);
                                            }
                                            let _e7503_ = phi_32421_;
                                            if _e7338_ {
                                                phi_32436_ = true;
                                            } else {
                                                phi_32436_ = (1f >= _e7306_);
                                            }
                                            let _e7507_ = phi_32436_;
                                            if _e7343_ {
                                                phi_32451_ = true;
                                            } else {
                                                phi_32451_ = (1f >= _e7311_);
                                            }
                                            let _e7511_ = phi_32451_;
                                            if _e7348_ {
                                                phi_32466_ = true;
                                            } else {
                                                phi_32466_ = (1f >= _e7316_);
                                            }
                                            let _e7515_ = phi_32466_;
                                            let _e7517_ = (_e7320_ - select(_e7301_, 1f, _e7503_));
                                            let _e7518_ = (_e7324_ - select(_e7306_, 1f, _e7507_));
                                            let _e7519_ = (_e7328_ - select(_e7311_, 1f, _e7511_));
                                            let _e7520_ = (_e7332_ - select(_e7316_, 1f, _e7515_));
                                            if (_e7517_ != _e7517_) {
                                                phi_32481_ = true;
                                            } else {
                                                phi_32481_ = (0f >= _e7517_);
                                            }
                                            let _e7524_ = phi_32481_;
                                            if (_e7518_ != _e7518_) {
                                                phi_32496_ = true;
                                            } else {
                                                phi_32496_ = (0f >= _e7518_);
                                            }
                                            let _e7529_ = phi_32496_;
                                            if (_e7519_ != _e7519_) {
                                                phi_32511_ = true;
                                            } else {
                                                phi_32511_ = (0f >= _e7519_);
                                            }
                                            let _e7534_ = phi_32511_;
                                            if (_e7520_ != _e7520_) {
                                                phi_32526_ = true;
                                            } else {
                                                phi_32526_ = (0f >= _e7520_);
                                            }
                                            let _e7539_ = phi_32526_;
                                            let _e7543_ = bitcast<f32>((bitcast<u32>(_e7220_) & 2147483647u));
                                            let _e7546_ = bitcast<f32>((bitcast<u32>(_e7225_) & 2147483647u));
                                            let _e7549_ = bitcast<f32>((bitcast<u32>(_e7230_) & 2147483647u));
                                            let _e7552_ = bitcast<f32>((bitcast<u32>(_e7235_) & 2147483647u));
                                            let _e7555_ = bitcast<f32>((bitcast<u32>(_e7280_) & 2147483647u));
                                            let _e7558_ = bitcast<f32>((bitcast<u32>(_e7285_) & 2147483647u));
                                            let _e7561_ = bitcast<f32>((bitcast<u32>(_e7290_) & 2147483647u));
                                            let _e7564_ = bitcast<f32>((bitcast<u32>(_e7295_) & 2147483647u));
                                            if (_e7543_ != _e7543_) {
                                                phi_32541_ = true;
                                            } else {
                                                phi_32541_ = (_e7555_ >= _e7543_);
                                            }
                                            let _e7568_ = phi_32541_;
                                            let _e7569_ = select(_e7543_, _e7555_, _e7568_);
                                            if (_e7546_ != _e7546_) {
                                                phi_32556_ = true;
                                            } else {
                                                phi_32556_ = (_e7558_ >= _e7546_);
                                            }
                                            let _e7573_ = phi_32556_;
                                            let _e7574_ = select(_e7546_, _e7558_, _e7573_);
                                            if (_e7549_ != _e7549_) {
                                                phi_32571_ = true;
                                            } else {
                                                phi_32571_ = (_e7561_ >= _e7549_);
                                            }
                                            let _e7578_ = phi_32571_;
                                            let _e7579_ = select(_e7549_, _e7561_, _e7578_);
                                            if (_e7552_ != _e7552_) {
                                                phi_32586_ = true;
                                            } else {
                                                phi_32586_ = (_e7564_ >= _e7552_);
                                            }
                                            let _e7583_ = phi_32586_;
                                            let _e7584_ = select(_e7552_, _e7564_, _e7583_);
                                            let _e7601_ = (_e7569_ <= 0f);
                                            let _e7602_ = select((_e7417_ * (1.5f - (0.5f * (_e7429_ + _e7433_)))), 0f, _e7601_);
                                            let _e7603_ = (_e7574_ <= 0f);
                                            let _e7604_ = select((_e7418_ * (1.5f - (0.5f * (_e7430_ + _e7434_)))), 0f, _e7603_);
                                            let _e7605_ = (_e7579_ <= 0f);
                                            let _e7606_ = select((_e7419_ * (1.5f - (0.5f * (_e7431_ + _e7435_)))), 0f, _e7605_);
                                            let _e7607_ = (_e7584_ <= 0f);
                                            let _e7608_ = select((_e7420_ * (1.5f - (0.5f * (_e7432_ + _e7436_)))), 0f, _e7607_);
                                            let _e7650_ = select(((_e7429_ + (0.33333334f * _e7433_)) - (((_e7429_ * _e7429_) + ((2f * _e7429_) * _e7433_)) + (0.2f * (_e7433_ * _e7433_)))), 0f, _e7601_);
                                            let _e7651_ = select(((_e7430_ + (0.33333334f * _e7434_)) - (((_e7430_ * _e7430_) + ((2f * _e7430_) * _e7434_)) + (0.2f * (_e7434_ * _e7434_)))), 0f, _e7603_);
                                            let _e7652_ = select(((_e7431_ + (0.33333334f * _e7435_)) - (((_e7431_ * _e7431_) + ((2f * _e7431_) * _e7435_)) + (0.2f * (_e7435_ * _e7435_)))), 0f, _e7605_);
                                            let _e7653_ = select(((_e7432_ + (0.33333334f * _e7436_)) - (((_e7432_ * _e7432_) + ((2f * _e7432_) * _e7436_)) + (0.2f * (_e7436_ * _e7436_)))), 0f, _e7607_);
                                            if (_e7569_ <= 1f) {
                                                if (_e7574_ <= 1f) {
                                                    let _e7657_ = (_e7579_ <= 1f);
                                                    if _e7657_ {
                                                        phi_20499_ = (_e7584_ <= 1f);
                                                    } else {
                                                        phi_20499_ = bool();
                                                    }
                                                    let _e7660_ = phi_20499_;
                                                    phi_20501_ = _e7660_;
                                                    phi_20502_ = select(true, false, _e7657_);
                                                } else {
                                                    phi_20501_ = bool();
                                                    phi_20502_ = true;
                                                }
                                                let _e7663_ = phi_20501_;
                                                let _e7665_ = phi_20502_;
                                                phi_20503_ = _e7663_;
                                                phi_20504_ = _e7665_;
                                            } else {
                                                phi_20503_ = bool();
                                                phi_20504_ = true;
                                            }
                                            let _e7667_ = phi_20503_;
                                            let _e7669_ = phi_20504_;
                                            if select(_e7667_, false, _e7669_) {
                                                phi_20882_ = type_37_(type_30_(array<f32, 4>(_e7602_, _e7604_, _e7606_, _e7608_)), type_30_(array<f32, 4>(_e7650_, _e7651_, _e7652_, _e7653_)), type_30_(_e7296_));
                                            } else {
                                                if (_e7457_ != _e7457_) {
                                                    phi_32601_ = true;
                                                } else {
                                                    phi_32601_ = (0.000001f >= _e7457_);
                                                }
                                                let _e7674_ = phi_32601_;
                                                let _e7675_ = select(_e7457_, 0.000001f, _e7674_);
                                                if (_e7458_ != _e7458_) {
                                                    phi_32616_ = true;
                                                } else {
                                                    phi_32616_ = (0.000001f >= _e7458_);
                                                }
                                                let _e7679_ = phi_32616_;
                                                let _e7680_ = select(_e7458_, 0.000001f, _e7679_);
                                                if (_e7459_ != _e7459_) {
                                                    phi_32631_ = true;
                                                } else {
                                                    phi_32631_ = (0.000001f >= _e7459_);
                                                }
                                                let _e7684_ = phi_32631_;
                                                let _e7685_ = select(_e7459_, 0.000001f, _e7684_);
                                                if (_e7460_ != _e7460_) {
                                                    phi_32646_ = true;
                                                } else {
                                                    phi_32646_ = (0.000001f >= _e7460_);
                                                }
                                                let _e7689_ = phi_32646_;
                                                let _e7690_ = select(_e7460_, 0.000001f, _e7689_);
                                                let _e7693_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7675_)));
                                                let _e7696_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7680_)));
                                                let _e7699_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7685_)));
                                                let _e7702_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7690_)));
                                                let _e7712_ = (_e7693_ + (_e7693_ * (1f - (_e7675_ * _e7693_))));
                                                let _e7714_ = (_e7696_ + (_e7696_ * (1f - (_e7680_ * _e7696_))));
                                                let _e7716_ = (_e7699_ + (_e7699_ * (1f - (_e7685_ * _e7699_))));
                                                let _e7718_ = (_e7702_ + (_e7702_ * (1f - (_e7690_ * _e7702_))));
                                                let _e7728_ = (_e7712_ + (_e7712_ * (1f - (_e7675_ * _e7712_))));
                                                let _e7730_ = (_e7714_ + (_e7714_ * (1f - (_e7680_ * _e7714_))));
                                                let _e7732_ = (_e7716_ + (_e7716_ * (1f - (_e7685_ * _e7716_))));
                                                let _e7734_ = (_e7718_ + (_e7718_ * (1f - (_e7690_ * _e7718_))));
                                                let _e7787_ = (select((select(0f, 1f, (-1f <= _e7301_)) * select(0f, 1f, (_e7320_ <= 1f))), 1f, (_e7457_ <= 0.000001f)) <= 0f);
                                                let _e7789_ = (select((select(0f, 1f, (-1f <= _e7306_)) * select(0f, 1f, (_e7324_ <= 1f))), 1f, (_e7458_ <= 0.000001f)) <= 0f);
                                                let _e7791_ = (select((select(0f, 1f, (-1f <= _e7311_)) * select(0f, 1f, (_e7328_ <= 1f))), 1f, (_e7459_ <= 0.000001f)) <= 0f);
                                                let _e7793_ = (select((select(0f, 1f, (-1f <= _e7316_)) * select(0f, 1f, (_e7332_ <= 1f))), 1f, (_e7460_ <= 0.000001f)) <= 0f);
                                                let _e7799_ = (1f <= _e7301_);
                                                let _e7801_ = (1f <= _e7306_);
                                                let _e7803_ = (1f <= _e7311_);
                                                let _e7805_ = (1f <= _e7316_);
                                                let _e7807_ = (_e7320_ <= -1f);
                                                let _e7809_ = (_e7324_ <= -1f);
                                                let _e7811_ = (_e7328_ <= -1f);
                                                let _e7813_ = (_e7332_ <= -1f);
                                                phi_20882_ = type_37_(type_30_(array<f32, 4>(select(select(select(_e7602_, (((select(_e7517_, 0f, _e7524_) - select(_e7477_, 0f, _e7484_)) + (_e7441_ * _e7602_)) * _e7728_), _e7787_), 1f, _e7799_), -1f, _e7807_), select(select(select(_e7604_, (((select(_e7518_, 0f, _e7529_) - select(_e7478_, 0f, _e7489_)) + (_e7446_ * _e7604_)) * _e7730_), _e7789_), 1f, _e7801_), -1f, _e7809_), select(select(select(_e7606_, (((select(_e7519_, 0f, _e7534_) - select(_e7479_, 0f, _e7494_)) + (_e7451_ * _e7606_)) * _e7732_), _e7791_), 1f, _e7803_), -1f, _e7811_), select(select(select(_e7608_, (((select(_e7520_, 0f, _e7539_) - select(_e7480_, 0f, _e7499_)) + (_e7456_ * _e7608_)) * _e7734_), _e7793_), 1f, _e7805_), -1f, _e7813_))), type_30_(array<f32, 4>(select(select(select(_e7650_, ((_e7441_ * _e7650_) * _e7728_), _e7787_), 0f, _e7799_), 0f, _e7807_), select(select(select(_e7651_, ((_e7446_ * _e7651_) * _e7730_), _e7789_), 0f, _e7801_), 0f, _e7809_), select(select(select(_e7652_, ((_e7451_ * _e7652_) * _e7732_), _e7791_), 0f, _e7803_), 0f, _e7811_), select(select(select(_e7653_, ((_e7456_ * _e7653_) * _e7734_), _e7793_), 0f, _e7805_), 0f, _e7813_))), type_30_(_e7296_));
                                            }
                                            let _e7834_ = phi_20882_;
                                            if (_e483_ <= 0f) {
                                                if (_e503_ <= 0f) {
                                                    let _e7838_ = (_e523_ <= 0f);
                                                    if _e7838_ {
                                                        phi_20899_ = (_e543_ <= 0f);
                                                    } else {
                                                        phi_20899_ = bool();
                                                    }
                                                    let _e7841_ = phi_20899_;
                                                    phi_20901_ = _e7841_;
                                                    phi_20902_ = select(true, false, _e7838_);
                                                } else {
                                                    phi_20901_ = bool();
                                                    phi_20902_ = true;
                                                }
                                                let _e7844_ = phi_20901_;
                                                let _e7846_ = phi_20902_;
                                                phi_20903_ = _e7844_;
                                                phi_20904_ = _e7846_;
                                            } else {
                                                phi_20903_ = bool();
                                                phi_20904_ = true;
                                            }
                                            let _e7848_ = phi_20903_;
                                            let _e7850_ = phi_20904_;
                                            if select(_e7848_, false, _e7850_) {
                                                phi_22812_ = _e7834_.member;
                                                phi_22813_ = _e7174_.member;
                                            } else {
                                                if _e6536_ {
                                                    phi_32661_ = true;
                                                } else {
                                                    phi_32661_ = (-1000000000000000000f >= _e6529_);
                                                }
                                                let _e7854_ = phi_32661_;
                                                let _e7855_ = select(_e6529_, -1000000000000000000f, _e7854_);
                                                if _e6541_ {
                                                    phi_32676_ = true;
                                                } else {
                                                    phi_32676_ = (-1000000000000000000f >= _e6531_);
                                                }
                                                let _e7858_ = phi_32676_;
                                                let _e7859_ = select(_e6531_, -1000000000000000000f, _e7858_);
                                                if _e6546_ {
                                                    phi_32691_ = true;
                                                } else {
                                                    phi_32691_ = (-1000000000000000000f >= _e6533_);
                                                }
                                                let _e7862_ = phi_32691_;
                                                let _e7863_ = select(_e6533_, -1000000000000000000f, _e7862_);
                                                if _e6551_ {
                                                    phi_32706_ = true;
                                                } else {
                                                    phi_32706_ = (-1000000000000000000f >= _e6535_);
                                                }
                                                let _e7866_ = phi_32706_;
                                                let _e7867_ = select(_e6535_, -1000000000000000000f, _e7866_);
                                                if (_e7855_ != _e7855_) {
                                                    phi_32721_ = true;
                                                } else {
                                                    phi_32721_ = (1000000000000000000f <= _e7855_);
                                                }
                                                let _e7871_ = phi_32721_;
                                                let _e7872_ = select(_e7855_, 1000000000000000000f, _e7871_);
                                                if (_e7859_ != _e7859_) {
                                                    phi_32736_ = true;
                                                } else {
                                                    phi_32736_ = (1000000000000000000f <= _e7859_);
                                                }
                                                let _e7876_ = phi_32736_;
                                                let _e7877_ = select(_e7859_, 1000000000000000000f, _e7876_);
                                                if (_e7863_ != _e7863_) {
                                                    phi_32751_ = true;
                                                } else {
                                                    phi_32751_ = (1000000000000000000f <= _e7863_);
                                                }
                                                let _e7881_ = phi_32751_;
                                                let _e7882_ = select(_e7863_, 1000000000000000000f, _e7881_);
                                                if (_e7867_ != _e7867_) {
                                                    phi_32766_ = true;
                                                } else {
                                                    phi_32766_ = (1000000000000000000f <= _e7867_);
                                                }
                                                let _e7886_ = phi_32766_;
                                                let _e7887_ = select(_e7867_, 1000000000000000000f, _e7886_);
                                                if _e6596_ {
                                                    phi_32781_ = true;
                                                } else {
                                                    phi_32781_ = (-1000000000000000000f >= _e6589_);
                                                }
                                                let _e7890_ = phi_32781_;
                                                let _e7891_ = select(_e6589_, -1000000000000000000f, _e7890_);
                                                if _e6601_ {
                                                    phi_32796_ = true;
                                                } else {
                                                    phi_32796_ = (-1000000000000000000f >= _e6591_);
                                                }
                                                let _e7894_ = phi_32796_;
                                                let _e7895_ = select(_e6591_, -1000000000000000000f, _e7894_);
                                                if _e6606_ {
                                                    phi_32811_ = true;
                                                } else {
                                                    phi_32811_ = (-1000000000000000000f >= _e6593_);
                                                }
                                                let _e7898_ = phi_32811_;
                                                let _e7899_ = select(_e6593_, -1000000000000000000f, _e7898_);
                                                if _e6611_ {
                                                    phi_32826_ = true;
                                                } else {
                                                    phi_32826_ = (-1000000000000000000f >= _e6595_);
                                                }
                                                let _e7902_ = phi_32826_;
                                                let _e7903_ = select(_e6595_, -1000000000000000000f, _e7902_);
                                                if (_e7891_ != _e7891_) {
                                                    phi_32841_ = true;
                                                } else {
                                                    phi_32841_ = (1000000000000000000f <= _e7891_);
                                                }
                                                let _e7907_ = phi_32841_;
                                                let _e7908_ = select(_e7891_, 1000000000000000000f, _e7907_);
                                                if (_e7895_ != _e7895_) {
                                                    phi_32856_ = true;
                                                } else {
                                                    phi_32856_ = (1000000000000000000f <= _e7895_);
                                                }
                                                let _e7912_ = phi_32856_;
                                                let _e7913_ = select(_e7895_, 1000000000000000000f, _e7912_);
                                                if (_e7899_ != _e7899_) {
                                                    phi_32871_ = true;
                                                } else {
                                                    phi_32871_ = (1000000000000000000f <= _e7899_);
                                                }
                                                let _e7917_ = phi_32871_;
                                                let _e7918_ = select(_e7899_, 1000000000000000000f, _e7917_);
                                                if (_e7903_ != _e7903_) {
                                                    phi_32886_ = true;
                                                } else {
                                                    phi_32886_ = (1000000000000000000f <= _e7903_);
                                                }
                                                let _e7922_ = phi_32886_;
                                                let _e7923_ = select(_e7903_, 1000000000000000000f, _e7922_);
                                                let _e7924_ = (_e7872_ != _e7872_);
                                                if _e7924_ {
                                                    phi_32901_ = true;
                                                } else {
                                                    phi_32901_ = (_e7908_ <= _e7872_);
                                                }
                                                let _e7927_ = phi_32901_;
                                                let _e7928_ = select(_e7872_, _e7908_, _e7927_);
                                                let _e7929_ = (_e7877_ != _e7877_);
                                                if _e7929_ {
                                                    phi_32916_ = true;
                                                } else {
                                                    phi_32916_ = (_e7913_ <= _e7877_);
                                                }
                                                let _e7932_ = phi_32916_;
                                                let _e7933_ = select(_e7877_, _e7913_, _e7932_);
                                                let _e7934_ = (_e7882_ != _e7882_);
                                                if _e7934_ {
                                                    phi_32931_ = true;
                                                } else {
                                                    phi_32931_ = (_e7918_ <= _e7882_);
                                                }
                                                let _e7937_ = phi_32931_;
                                                let _e7938_ = select(_e7882_, _e7918_, _e7937_);
                                                let _e7939_ = (_e7887_ != _e7887_);
                                                if _e7939_ {
                                                    phi_32946_ = true;
                                                } else {
                                                    phi_32946_ = (_e7923_ <= _e7887_);
                                                }
                                                let _e7942_ = phi_32946_;
                                                let _e7943_ = select(_e7887_, _e7923_, _e7942_);
                                                if _e7924_ {
                                                    phi_32961_ = true;
                                                } else {
                                                    phi_32961_ = (_e7908_ >= _e7872_);
                                                }
                                                let _e7946_ = phi_32961_;
                                                let _e7947_ = select(_e7872_, _e7908_, _e7946_);
                                                if _e7929_ {
                                                    phi_32976_ = true;
                                                } else {
                                                    phi_32976_ = (_e7913_ >= _e7877_);
                                                }
                                                let _e7950_ = phi_32976_;
                                                let _e7951_ = select(_e7877_, _e7913_, _e7950_);
                                                if _e7934_ {
                                                    phi_32991_ = true;
                                                } else {
                                                    phi_32991_ = (_e7918_ >= _e7882_);
                                                }
                                                let _e7954_ = phi_32991_;
                                                let _e7955_ = select(_e7882_, _e7918_, _e7954_);
                                                if _e7939_ {
                                                    phi_33006_ = true;
                                                } else {
                                                    phi_33006_ = (_e7923_ >= _e7887_);
                                                }
                                                let _e7958_ = phi_33006_;
                                                let _e7959_ = select(_e7887_, _e7923_, _e7958_);
                                                if (_e7928_ != _e7928_) {
                                                    phi_33021_ = true;
                                                } else {
                                                    phi_33021_ = (-1f >= _e7928_);
                                                }
                                                let _e7963_ = phi_33021_;
                                                let _e7964_ = select(_e7928_, -1f, _e7963_);
                                                if (_e7933_ != _e7933_) {
                                                    phi_33036_ = true;
                                                } else {
                                                    phi_33036_ = (-1f >= _e7933_);
                                                }
                                                let _e7968_ = phi_33036_;
                                                let _e7969_ = select(_e7933_, -1f, _e7968_);
                                                if (_e7938_ != _e7938_) {
                                                    phi_33051_ = true;
                                                } else {
                                                    phi_33051_ = (-1f >= _e7938_);
                                                }
                                                let _e7973_ = phi_33051_;
                                                let _e7974_ = select(_e7938_, -1f, _e7973_);
                                                if (_e7943_ != _e7943_) {
                                                    phi_33066_ = true;
                                                } else {
                                                    phi_33066_ = (-1f >= _e7943_);
                                                }
                                                let _e7978_ = phi_33066_;
                                                let _e7979_ = select(_e7943_, -1f, _e7978_);
                                                if (_e7964_ != _e7964_) {
                                                    phi_33081_ = true;
                                                } else {
                                                    phi_33081_ = (1f <= _e7964_);
                                                }
                                                let _e7983_ = phi_33081_;
                                                let _e7984_ = select(_e7964_, 1f, _e7983_);
                                                if (_e7969_ != _e7969_) {
                                                    phi_33096_ = true;
                                                } else {
                                                    phi_33096_ = (1f <= _e7969_);
                                                }
                                                let _e7988_ = phi_33096_;
                                                let _e7989_ = select(_e7969_, 1f, _e7988_);
                                                if (_e7974_ != _e7974_) {
                                                    phi_33111_ = true;
                                                } else {
                                                    phi_33111_ = (1f <= _e7974_);
                                                }
                                                let _e7993_ = phi_33111_;
                                                let _e7994_ = select(_e7974_, 1f, _e7993_);
                                                if (_e7979_ != _e7979_) {
                                                    phi_33126_ = true;
                                                } else {
                                                    phi_33126_ = (1f <= _e7979_);
                                                }
                                                let _e7998_ = phi_33126_;
                                                let _e7999_ = select(_e7979_, 1f, _e7998_);
                                                if (_e7947_ != _e7947_) {
                                                    phi_33141_ = true;
                                                } else {
                                                    phi_33141_ = (-1f >= _e7947_);
                                                }
                                                let _e8003_ = phi_33141_;
                                                let _e8004_ = select(_e7947_, -1f, _e8003_);
                                                if (_e7951_ != _e7951_) {
                                                    phi_33156_ = true;
                                                } else {
                                                    phi_33156_ = (-1f >= _e7951_);
                                                }
                                                let _e8008_ = phi_33156_;
                                                let _e8009_ = select(_e7951_, -1f, _e8008_);
                                                if (_e7955_ != _e7955_) {
                                                    phi_33171_ = true;
                                                } else {
                                                    phi_33171_ = (-1f >= _e7955_);
                                                }
                                                let _e8013_ = phi_33171_;
                                                let _e8014_ = select(_e7955_, -1f, _e8013_);
                                                if (_e7959_ != _e7959_) {
                                                    phi_33186_ = true;
                                                } else {
                                                    phi_33186_ = (-1f >= _e7959_);
                                                }
                                                let _e8018_ = phi_33186_;
                                                let _e8019_ = select(_e7959_, -1f, _e8018_);
                                                if (_e8004_ != _e8004_) {
                                                    phi_33201_ = true;
                                                } else {
                                                    phi_33201_ = (1f <= _e8004_);
                                                }
                                                let _e8023_ = phi_33201_;
                                                let _e8024_ = select(_e8004_, 1f, _e8023_);
                                                if (_e8009_ != _e8009_) {
                                                    phi_33216_ = true;
                                                } else {
                                                    phi_33216_ = (1f <= _e8009_);
                                                }
                                                let _e8028_ = phi_33216_;
                                                let _e8029_ = select(_e8009_, 1f, _e8028_);
                                                if (_e8014_ != _e8014_) {
                                                    phi_33231_ = true;
                                                } else {
                                                    phi_33231_ = (1f <= _e8014_);
                                                }
                                                let _e8033_ = phi_33231_;
                                                let _e8034_ = select(_e8014_, 1f, _e8033_);
                                                if (_e8019_ != _e8019_) {
                                                    phi_33246_ = true;
                                                } else {
                                                    phi_33246_ = (1f <= _e8019_);
                                                }
                                                let _e8038_ = phi_33246_;
                                                let _e8039_ = select(_e8019_, 1f, _e8038_);
                                                let _e8044_ = (0.5f * (_e8024_ + _e7984_));
                                                let _e8045_ = (0.5f * (_e8029_ + _e7989_));
                                                let _e8046_ = (0.5f * (_e8034_ + _e7994_));
                                                let _e8047_ = (0.5f * (_e8039_ + _e7999_));
                                                let _e8048_ = (_e8024_ - _e7984_);
                                                let _e8049_ = (_e8029_ - _e7989_);
                                                let _e8050_ = (_e8034_ - _e7994_);
                                                let _e8051_ = (_e8039_ - _e7999_);
                                                let _e8052_ = (0.5f * _e8048_);
                                                let _e8053_ = (0.5f * _e8049_);
                                                let _e8054_ = (0.5f * _e8050_);
                                                let _e8055_ = (0.5f * _e8051_);
                                                let _e8056_ = (_e8044_ * _e8044_);
                                                let _e8057_ = (_e8045_ * _e8045_);
                                                let _e8058_ = (_e8046_ * _e8046_);
                                                let _e8059_ = (_e8047_ * _e8047_);
                                                let _e8060_ = (_e8052_ * _e8052_);
                                                let _e8061_ = (_e8053_ * _e8053_);
                                                let _e8062_ = (_e8054_ * _e8054_);
                                                let _e8063_ = (_e8055_ * _e8055_);
                                                if (_e8048_ != _e8048_) {
                                                    phi_33261_ = true;
                                                } else {
                                                    phi_33261_ = (0f >= _e8048_);
                                                }
                                                let _e8067_ = phi_33261_;
                                                if (_e8049_ != _e8049_) {
                                                    phi_33276_ = true;
                                                } else {
                                                    phi_33276_ = (0f >= _e8049_);
                                                }
                                                let _e8072_ = phi_33276_;
                                                if (_e8050_ != _e8050_) {
                                                    phi_33291_ = true;
                                                } else {
                                                    phi_33291_ = (0f >= _e8050_);
                                                }
                                                let _e8077_ = phi_33291_;
                                                if (_e8051_ != _e8051_) {
                                                    phi_33306_ = true;
                                                } else {
                                                    phi_33306_ = (0f >= _e8051_);
                                                }
                                                let _e8082_ = phi_33306_;
                                                let _e8084_ = (_e7947_ - _e7928_);
                                                let _e8085_ = (_e7951_ - _e7933_);
                                                let _e8086_ = (_e7955_ - _e7938_);
                                                let _e8087_ = (_e7959_ - _e7943_);
                                                let _e8090_ = bitcast<f32>((bitcast<u32>(_e7872_) & 2147483647u));
                                                let _e8093_ = bitcast<f32>((bitcast<u32>(_e7877_) & 2147483647u));
                                                let _e8096_ = bitcast<f32>((bitcast<u32>(_e7882_) & 2147483647u));
                                                let _e8099_ = bitcast<f32>((bitcast<u32>(_e7887_) & 2147483647u));
                                                let _e8102_ = bitcast<f32>((bitcast<u32>(_e7908_) & 2147483647u));
                                                let _e8105_ = bitcast<f32>((bitcast<u32>(_e7913_) & 2147483647u));
                                                let _e8108_ = bitcast<f32>((bitcast<u32>(_e7918_) & 2147483647u));
                                                let _e8111_ = bitcast<f32>((bitcast<u32>(_e7923_) & 2147483647u));
                                                if (_e8090_ != _e8090_) {
                                                    phi_33561_ = true;
                                                } else {
                                                    phi_33561_ = (_e8102_ >= _e8090_);
                                                }
                                                let _e8115_ = phi_33561_;
                                                if (_e8093_ != _e8093_) {
                                                    phi_33576_ = true;
                                                } else {
                                                    phi_33576_ = (_e8105_ >= _e8093_);
                                                }
                                                let _e8120_ = phi_33576_;
                                                if (_e8096_ != _e8096_) {
                                                    phi_33591_ = true;
                                                } else {
                                                    phi_33591_ = (_e8108_ >= _e8096_);
                                                }
                                                let _e8125_ = phi_33591_;
                                                if (_e8099_ != _e8099_) {
                                                    phi_33606_ = true;
                                                } else {
                                                    phi_33606_ = (_e8111_ >= _e8099_);
                                                }
                                                let _e8130_ = phi_33606_;
                                                let _e8136_ = (1f - (_e8056_ + _e8060_));
                                                let _e8137_ = (1f - (_e8057_ + _e8061_));
                                                let _e8138_ = (1f - (_e8058_ + _e8062_));
                                                let _e8139_ = (1f - (_e8059_ + _e8063_));
                                                let _e8165_ = select(((_e8044_ * _e8136_) * ((_e8136_ * _e8136_) + ((4f * _e8056_) * _e8060_))), 0f, (select(_e8090_, _e8102_, _e8115_) <= 0f));
                                                let _e8167_ = select(((_e8045_ * _e8137_) * ((_e8137_ * _e8137_) + ((4f * _e8057_) * _e8061_))), 0f, (select(_e8093_, _e8105_, _e8120_) <= 0f));
                                                let _e8169_ = select(((_e8046_ * _e8138_) * ((_e8138_ * _e8138_) + ((4f * _e8058_) * _e8062_))), 0f, (select(_e8096_, _e8108_, _e8125_) <= 0f));
                                                let _e8171_ = select(((_e8047_ * _e8139_) * ((_e8139_ * _e8139_) + ((4f * _e8059_) * _e8063_))), 0f, (select(_e8099_, _e8111_, _e8130_) <= 0f));
                                                if (_e8084_ != _e8084_) {
                                                    phi_33621_ = true;
                                                } else {
                                                    phi_33621_ = (0.000001f >= _e8084_);
                                                }
                                                let _e8175_ = phi_33621_;
                                                let _e8176_ = select(_e8084_, 0.000001f, _e8175_);
                                                if (_e8085_ != _e8085_) {
                                                    phi_33636_ = true;
                                                } else {
                                                    phi_33636_ = (0.000001f >= _e8085_);
                                                }
                                                let _e8180_ = phi_33636_;
                                                let _e8181_ = select(_e8085_, 0.000001f, _e8180_);
                                                if (_e8086_ != _e8086_) {
                                                    phi_33651_ = true;
                                                } else {
                                                    phi_33651_ = (0.000001f >= _e8086_);
                                                }
                                                let _e8185_ = phi_33651_;
                                                let _e8186_ = select(_e8086_, 0.000001f, _e8185_);
                                                if (_e8087_ != _e8087_) {
                                                    phi_33666_ = true;
                                                } else {
                                                    phi_33666_ = (0.000001f >= _e8087_);
                                                }
                                                let _e8190_ = phi_33666_;
                                                let _e8191_ = select(_e8087_, 0.000001f, _e8190_);
                                                let _e8194_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8176_)));
                                                let _e8197_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8181_)));
                                                let _e8200_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8186_)));
                                                let _e8203_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8191_)));
                                                let _e8213_ = (_e8194_ + (_e8194_ * (1f - (_e8176_ * _e8194_))));
                                                let _e8215_ = (_e8197_ + (_e8197_ * (1f - (_e8181_ * _e8197_))));
                                                let _e8217_ = (_e8200_ + (_e8200_ * (1f - (_e8186_ * _e8200_))));
                                                let _e8219_ = (_e8203_ + (_e8203_ * (1f - (_e8191_ * _e8203_))));
                                                if _e7196_ {
                                                    phi_33681_ = true;
                                                } else {
                                                    phi_33681_ = (-1000000000000000000f >= _e7189_);
                                                }
                                                let _e8298_ = phi_33681_;
                                                let _e8299_ = select(_e7189_, -1000000000000000000f, _e8298_);
                                                if _e7201_ {
                                                    phi_33696_ = true;
                                                } else {
                                                    phi_33696_ = (-1000000000000000000f >= _e7191_);
                                                }
                                                let _e8302_ = phi_33696_;
                                                let _e8303_ = select(_e7191_, -1000000000000000000f, _e8302_);
                                                if _e7206_ {
                                                    phi_33711_ = true;
                                                } else {
                                                    phi_33711_ = (-1000000000000000000f >= _e7193_);
                                                }
                                                let _e8306_ = phi_33711_;
                                                let _e8307_ = select(_e7193_, -1000000000000000000f, _e8306_);
                                                if _e7211_ {
                                                    phi_33726_ = true;
                                                } else {
                                                    phi_33726_ = (-1000000000000000000f >= _e7195_);
                                                }
                                                let _e8310_ = phi_33726_;
                                                let _e8311_ = select(_e7195_, -1000000000000000000f, _e8310_);
                                                if (_e8299_ != _e8299_) {
                                                    phi_33741_ = true;
                                                } else {
                                                    phi_33741_ = (1000000000000000000f <= _e8299_);
                                                }
                                                let _e8315_ = phi_33741_;
                                                let _e8316_ = select(_e8299_, 1000000000000000000f, _e8315_);
                                                if (_e8303_ != _e8303_) {
                                                    phi_33756_ = true;
                                                } else {
                                                    phi_33756_ = (1000000000000000000f <= _e8303_);
                                                }
                                                let _e8320_ = phi_33756_;
                                                let _e8321_ = select(_e8303_, 1000000000000000000f, _e8320_);
                                                if (_e8307_ != _e8307_) {
                                                    phi_33771_ = true;
                                                } else {
                                                    phi_33771_ = (1000000000000000000f <= _e8307_);
                                                }
                                                let _e8325_ = phi_33771_;
                                                let _e8326_ = select(_e8307_, 1000000000000000000f, _e8325_);
                                                if (_e8311_ != _e8311_) {
                                                    phi_33786_ = true;
                                                } else {
                                                    phi_33786_ = (1000000000000000000f <= _e8311_);
                                                }
                                                let _e8330_ = phi_33786_;
                                                let _e8331_ = select(_e8311_, 1000000000000000000f, _e8330_);
                                                if _e7256_ {
                                                    phi_33801_ = true;
                                                } else {
                                                    phi_33801_ = (-1000000000000000000f >= _e7249_);
                                                }
                                                let _e8334_ = phi_33801_;
                                                let _e8335_ = select(_e7249_, -1000000000000000000f, _e8334_);
                                                if _e7261_ {
                                                    phi_33816_ = true;
                                                } else {
                                                    phi_33816_ = (-1000000000000000000f >= _e7251_);
                                                }
                                                let _e8338_ = phi_33816_;
                                                let _e8339_ = select(_e7251_, -1000000000000000000f, _e8338_);
                                                if _e7266_ {
                                                    phi_33831_ = true;
                                                } else {
                                                    phi_33831_ = (-1000000000000000000f >= _e7253_);
                                                }
                                                let _e8342_ = phi_33831_;
                                                let _e8343_ = select(_e7253_, -1000000000000000000f, _e8342_);
                                                if _e7271_ {
                                                    phi_33846_ = true;
                                                } else {
                                                    phi_33846_ = (-1000000000000000000f >= _e7255_);
                                                }
                                                let _e8346_ = phi_33846_;
                                                let _e8347_ = select(_e7255_, -1000000000000000000f, _e8346_);
                                                if (_e8335_ != _e8335_) {
                                                    phi_33861_ = true;
                                                } else {
                                                    phi_33861_ = (1000000000000000000f <= _e8335_);
                                                }
                                                let _e8351_ = phi_33861_;
                                                let _e8352_ = select(_e8335_, 1000000000000000000f, _e8351_);
                                                if (_e8339_ != _e8339_) {
                                                    phi_33876_ = true;
                                                } else {
                                                    phi_33876_ = (1000000000000000000f <= _e8339_);
                                                }
                                                let _e8356_ = phi_33876_;
                                                let _e8357_ = select(_e8339_, 1000000000000000000f, _e8356_);
                                                if (_e8343_ != _e8343_) {
                                                    phi_33891_ = true;
                                                } else {
                                                    phi_33891_ = (1000000000000000000f <= _e8343_);
                                                }
                                                let _e8361_ = phi_33891_;
                                                let _e8362_ = select(_e8343_, 1000000000000000000f, _e8361_);
                                                if (_e8347_ != _e8347_) {
                                                    phi_33906_ = true;
                                                } else {
                                                    phi_33906_ = (1000000000000000000f <= _e8347_);
                                                }
                                                let _e8366_ = phi_33906_;
                                                let _e8367_ = select(_e8347_, 1000000000000000000f, _e8366_);
                                                let _e8368_ = (_e8316_ != _e8316_);
                                                if _e8368_ {
                                                    phi_33921_ = true;
                                                } else {
                                                    phi_33921_ = (_e8352_ <= _e8316_);
                                                }
                                                let _e8371_ = phi_33921_;
                                                let _e8372_ = select(_e8316_, _e8352_, _e8371_);
                                                let _e8373_ = (_e8321_ != _e8321_);
                                                if _e8373_ {
                                                    phi_33936_ = true;
                                                } else {
                                                    phi_33936_ = (_e8357_ <= _e8321_);
                                                }
                                                let _e8376_ = phi_33936_;
                                                let _e8377_ = select(_e8321_, _e8357_, _e8376_);
                                                let _e8378_ = (_e8326_ != _e8326_);
                                                if _e8378_ {
                                                    phi_33951_ = true;
                                                } else {
                                                    phi_33951_ = (_e8362_ <= _e8326_);
                                                }
                                                let _e8381_ = phi_33951_;
                                                let _e8382_ = select(_e8326_, _e8362_, _e8381_);
                                                let _e8383_ = (_e8331_ != _e8331_);
                                                if _e8383_ {
                                                    phi_33966_ = true;
                                                } else {
                                                    phi_33966_ = (_e8367_ <= _e8331_);
                                                }
                                                let _e8386_ = phi_33966_;
                                                let _e8387_ = select(_e8331_, _e8367_, _e8386_);
                                                if _e8368_ {
                                                    phi_33981_ = true;
                                                } else {
                                                    phi_33981_ = (_e8352_ >= _e8316_);
                                                }
                                                let _e8390_ = phi_33981_;
                                                let _e8391_ = select(_e8316_, _e8352_, _e8390_);
                                                if _e8373_ {
                                                    phi_33996_ = true;
                                                } else {
                                                    phi_33996_ = (_e8357_ >= _e8321_);
                                                }
                                                let _e8394_ = phi_33996_;
                                                let _e8395_ = select(_e8321_, _e8357_, _e8394_);
                                                if _e8378_ {
                                                    phi_34011_ = true;
                                                } else {
                                                    phi_34011_ = (_e8362_ >= _e8326_);
                                                }
                                                let _e8398_ = phi_34011_;
                                                let _e8399_ = select(_e8326_, _e8362_, _e8398_);
                                                if _e8383_ {
                                                    phi_34026_ = true;
                                                } else {
                                                    phi_34026_ = (_e8367_ >= _e8331_);
                                                }
                                                let _e8402_ = phi_34026_;
                                                let _e8403_ = select(_e8331_, _e8367_, _e8402_);
                                                if (_e8372_ != _e8372_) {
                                                    phi_34041_ = true;
                                                } else {
                                                    phi_34041_ = (-1f >= _e8372_);
                                                }
                                                let _e8407_ = phi_34041_;
                                                let _e8408_ = select(_e8372_, -1f, _e8407_);
                                                if (_e8377_ != _e8377_) {
                                                    phi_34056_ = true;
                                                } else {
                                                    phi_34056_ = (-1f >= _e8377_);
                                                }
                                                let _e8412_ = phi_34056_;
                                                let _e8413_ = select(_e8377_, -1f, _e8412_);
                                                if (_e8382_ != _e8382_) {
                                                    phi_34071_ = true;
                                                } else {
                                                    phi_34071_ = (-1f >= _e8382_);
                                                }
                                                let _e8417_ = phi_34071_;
                                                let _e8418_ = select(_e8382_, -1f, _e8417_);
                                                if (_e8387_ != _e8387_) {
                                                    phi_34086_ = true;
                                                } else {
                                                    phi_34086_ = (-1f >= _e8387_);
                                                }
                                                let _e8422_ = phi_34086_;
                                                let _e8423_ = select(_e8387_, -1f, _e8422_);
                                                if (_e8408_ != _e8408_) {
                                                    phi_34101_ = true;
                                                } else {
                                                    phi_34101_ = (1f <= _e8408_);
                                                }
                                                let _e8427_ = phi_34101_;
                                                let _e8428_ = select(_e8408_, 1f, _e8427_);
                                                if (_e8413_ != _e8413_) {
                                                    phi_34116_ = true;
                                                } else {
                                                    phi_34116_ = (1f <= _e8413_);
                                                }
                                                let _e8432_ = phi_34116_;
                                                let _e8433_ = select(_e8413_, 1f, _e8432_);
                                                if (_e8418_ != _e8418_) {
                                                    phi_34131_ = true;
                                                } else {
                                                    phi_34131_ = (1f <= _e8418_);
                                                }
                                                let _e8437_ = phi_34131_;
                                                let _e8438_ = select(_e8418_, 1f, _e8437_);
                                                if (_e8423_ != _e8423_) {
                                                    phi_34146_ = true;
                                                } else {
                                                    phi_34146_ = (1f <= _e8423_);
                                                }
                                                let _e8442_ = phi_34146_;
                                                let _e8443_ = select(_e8423_, 1f, _e8442_);
                                                if (_e8391_ != _e8391_) {
                                                    phi_34161_ = true;
                                                } else {
                                                    phi_34161_ = (-1f >= _e8391_);
                                                }
                                                let _e8447_ = phi_34161_;
                                                let _e8448_ = select(_e8391_, -1f, _e8447_);
                                                if (_e8395_ != _e8395_) {
                                                    phi_34176_ = true;
                                                } else {
                                                    phi_34176_ = (-1f >= _e8395_);
                                                }
                                                let _e8452_ = phi_34176_;
                                                let _e8453_ = select(_e8395_, -1f, _e8452_);
                                                if (_e8399_ != _e8399_) {
                                                    phi_34191_ = true;
                                                } else {
                                                    phi_34191_ = (-1f >= _e8399_);
                                                }
                                                let _e8457_ = phi_34191_;
                                                let _e8458_ = select(_e8399_, -1f, _e8457_);
                                                if (_e8403_ != _e8403_) {
                                                    phi_34206_ = true;
                                                } else {
                                                    phi_34206_ = (-1f >= _e8403_);
                                                }
                                                let _e8462_ = phi_34206_;
                                                let _e8463_ = select(_e8403_, -1f, _e8462_);
                                                if (_e8448_ != _e8448_) {
                                                    phi_34221_ = true;
                                                } else {
                                                    phi_34221_ = (1f <= _e8448_);
                                                }
                                                let _e8467_ = phi_34221_;
                                                let _e8468_ = select(_e8448_, 1f, _e8467_);
                                                if (_e8453_ != _e8453_) {
                                                    phi_34236_ = true;
                                                } else {
                                                    phi_34236_ = (1f <= _e8453_);
                                                }
                                                let _e8472_ = phi_34236_;
                                                let _e8473_ = select(_e8453_, 1f, _e8472_);
                                                if (_e8458_ != _e8458_) {
                                                    phi_34251_ = true;
                                                } else {
                                                    phi_34251_ = (1f <= _e8458_);
                                                }
                                                let _e8477_ = phi_34251_;
                                                let _e8478_ = select(_e8458_, 1f, _e8477_);
                                                if (_e8463_ != _e8463_) {
                                                    phi_34266_ = true;
                                                } else {
                                                    phi_34266_ = (1f <= _e8463_);
                                                }
                                                let _e8482_ = phi_34266_;
                                                let _e8483_ = select(_e8463_, 1f, _e8482_);
                                                let _e8488_ = (0.5f * (_e8468_ + _e8428_));
                                                let _e8489_ = (0.5f * (_e8473_ + _e8433_));
                                                let _e8490_ = (0.5f * (_e8478_ + _e8438_));
                                                let _e8491_ = (0.5f * (_e8483_ + _e8443_));
                                                let _e8492_ = (_e8468_ - _e8428_);
                                                let _e8493_ = (_e8473_ - _e8433_);
                                                let _e8494_ = (_e8478_ - _e8438_);
                                                let _e8495_ = (_e8483_ - _e8443_);
                                                let _e8496_ = (0.5f * _e8492_);
                                                let _e8497_ = (0.5f * _e8493_);
                                                let _e8498_ = (0.5f * _e8494_);
                                                let _e8499_ = (0.5f * _e8495_);
                                                let _e8500_ = (_e8488_ * _e8488_);
                                                let _e8501_ = (_e8489_ * _e8489_);
                                                let _e8502_ = (_e8490_ * _e8490_);
                                                let _e8503_ = (_e8491_ * _e8491_);
                                                let _e8504_ = (_e8496_ * _e8496_);
                                                let _e8505_ = (_e8497_ * _e8497_);
                                                let _e8506_ = (_e8498_ * _e8498_);
                                                let _e8507_ = (_e8499_ * _e8499_);
                                                if (_e8492_ != _e8492_) {
                                                    phi_34281_ = true;
                                                } else {
                                                    phi_34281_ = (0f >= _e8492_);
                                                }
                                                let _e8511_ = phi_34281_;
                                                if (_e8493_ != _e8493_) {
                                                    phi_34296_ = true;
                                                } else {
                                                    phi_34296_ = (0f >= _e8493_);
                                                }
                                                let _e8516_ = phi_34296_;
                                                if (_e8494_ != _e8494_) {
                                                    phi_34311_ = true;
                                                } else {
                                                    phi_34311_ = (0f >= _e8494_);
                                                }
                                                let _e8521_ = phi_34311_;
                                                if (_e8495_ != _e8495_) {
                                                    phi_34326_ = true;
                                                } else {
                                                    phi_34326_ = (0f >= _e8495_);
                                                }
                                                let _e8526_ = phi_34326_;
                                                let _e8528_ = (_e8391_ - _e8372_);
                                                let _e8529_ = (_e8395_ - _e8377_);
                                                let _e8530_ = (_e8399_ - _e8382_);
                                                let _e8531_ = (_e8403_ - _e8387_);
                                                let _e8534_ = bitcast<f32>((bitcast<u32>(_e8316_) & 2147483647u));
                                                let _e8537_ = bitcast<f32>((bitcast<u32>(_e8321_) & 2147483647u));
                                                let _e8540_ = bitcast<f32>((bitcast<u32>(_e8326_) & 2147483647u));
                                                let _e8543_ = bitcast<f32>((bitcast<u32>(_e8331_) & 2147483647u));
                                                let _e8546_ = bitcast<f32>((bitcast<u32>(_e8352_) & 2147483647u));
                                                let _e8549_ = bitcast<f32>((bitcast<u32>(_e8357_) & 2147483647u));
                                                let _e8552_ = bitcast<f32>((bitcast<u32>(_e8362_) & 2147483647u));
                                                let _e8555_ = bitcast<f32>((bitcast<u32>(_e8367_) & 2147483647u));
                                                if (_e8534_ != _e8534_) {
                                                    phi_34581_ = true;
                                                } else {
                                                    phi_34581_ = (_e8546_ >= _e8534_);
                                                }
                                                let _e8559_ = phi_34581_;
                                                if (_e8537_ != _e8537_) {
                                                    phi_34596_ = true;
                                                } else {
                                                    phi_34596_ = (_e8549_ >= _e8537_);
                                                }
                                                let _e8564_ = phi_34596_;
                                                if (_e8540_ != _e8540_) {
                                                    phi_34611_ = true;
                                                } else {
                                                    phi_34611_ = (_e8552_ >= _e8540_);
                                                }
                                                let _e8569_ = phi_34611_;
                                                if (_e8543_ != _e8543_) {
                                                    phi_34626_ = true;
                                                } else {
                                                    phi_34626_ = (_e8555_ >= _e8543_);
                                                }
                                                let _e8574_ = phi_34626_;
                                                let _e8580_ = (1f - (_e8500_ + _e8504_));
                                                let _e8581_ = (1f - (_e8501_ + _e8505_));
                                                let _e8582_ = (1f - (_e8502_ + _e8506_));
                                                let _e8583_ = (1f - (_e8503_ + _e8507_));
                                                let _e8609_ = select(((_e8488_ * _e8580_) * ((_e8580_ * _e8580_) + ((4f * _e8500_) * _e8504_))), 0f, (select(_e8534_, _e8546_, _e8559_) <= 0f));
                                                let _e8611_ = select(((_e8489_ * _e8581_) * ((_e8581_ * _e8581_) + ((4f * _e8501_) * _e8505_))), 0f, (select(_e8537_, _e8549_, _e8564_) <= 0f));
                                                let _e8613_ = select(((_e8490_ * _e8582_) * ((_e8582_ * _e8582_) + ((4f * _e8502_) * _e8506_))), 0f, (select(_e8540_, _e8552_, _e8569_) <= 0f));
                                                let _e8615_ = select(((_e8491_ * _e8583_) * ((_e8583_ * _e8583_) + ((4f * _e8503_) * _e8507_))), 0f, (select(_e8543_, _e8555_, _e8574_) <= 0f));
                                                if (_e8528_ != _e8528_) {
                                                    phi_34641_ = true;
                                                } else {
                                                    phi_34641_ = (0.000001f >= _e8528_);
                                                }
                                                let _e8619_ = phi_34641_;
                                                let _e8620_ = select(_e8528_, 0.000001f, _e8619_);
                                                if (_e8529_ != _e8529_) {
                                                    phi_34656_ = true;
                                                } else {
                                                    phi_34656_ = (0.000001f >= _e8529_);
                                                }
                                                let _e8624_ = phi_34656_;
                                                let _e8625_ = select(_e8529_, 0.000001f, _e8624_);
                                                if (_e8530_ != _e8530_) {
                                                    phi_34671_ = true;
                                                } else {
                                                    phi_34671_ = (0.000001f >= _e8530_);
                                                }
                                                let _e8629_ = phi_34671_;
                                                let _e8630_ = select(_e8530_, 0.000001f, _e8629_);
                                                if (_e8531_ != _e8531_) {
                                                    phi_34686_ = true;
                                                } else {
                                                    phi_34686_ = (0.000001f >= _e8531_);
                                                }
                                                let _e8634_ = phi_34686_;
                                                let _e8635_ = select(_e8531_, 0.000001f, _e8634_);
                                                let _e8638_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8620_)));
                                                let _e8641_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8625_)));
                                                let _e8644_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8630_)));
                                                let _e8647_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8635_)));
                                                let _e8657_ = (_e8638_ + (_e8638_ * (1f - (_e8620_ * _e8638_))));
                                                let _e8659_ = (_e8641_ + (_e8641_ * (1f - (_e8625_ * _e8641_))));
                                                let _e8661_ = (_e8644_ + (_e8644_ * (1f - (_e8630_ * _e8644_))));
                                                let _e8663_ = (_e8647_ + (_e8647_ * (1f - (_e8635_ * _e8647_))));
                                                let _e8740_ = (1f - _e483_);
                                                let _e8741_ = (1f - _e503_);
                                                let _e8742_ = (1f - _e523_);
                                                let _e8743_ = (1f - _e543_);
                                                phi_22812_ = type_30_(array<f32, 4>(((_e7834_.member.member[0] * _e8740_) + (_e483_ * select(select(select(_e8609_, ((select(_e8492_, 0f, _e8511_) * _e8609_) * (_e8657_ + (_e8657_ * (1f - (_e8620_ * _e8657_))))), (select((select(0f, 1f, (-1f <= _e8372_)) * select(0f, 1f, (_e8391_ <= 1f))), 1f, (_e8528_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8372_)), 0f, (_e8391_ <= -1f)))), ((_e7834_.member.member[1] * _e8741_) + (_e503_ * select(select(select(_e8611_, ((select(_e8493_, 0f, _e8516_) * _e8611_) * (_e8659_ + (_e8659_ * (1f - (_e8625_ * _e8659_))))), (select((select(0f, 1f, (-1f <= _e8377_)) * select(0f, 1f, (_e8395_ <= 1f))), 1f, (_e8529_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8377_)), 0f, (_e8395_ <= -1f)))), ((_e7834_.member.member[2] * _e8742_) + (_e523_ * select(select(select(_e8613_, ((select(_e8494_, 0f, _e8521_) * _e8613_) * (_e8661_ + (_e8661_ * (1f - (_e8630_ * _e8661_))))), (select((select(0f, 1f, (-1f <= _e8382_)) * select(0f, 1f, (_e8399_ <= 1f))), 1f, (_e8530_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8382_)), 0f, (_e8399_ <= -1f)))), ((_e7834_.member.member[3] * _e8743_) + (_e543_ * select(select(select(_e8615_, ((select(_e8495_, 0f, _e8526_) * _e8615_) * (_e8663_ + (_e8663_ * (1f - (_e8635_ * _e8663_))))), (select((select(0f, 1f, (-1f <= _e8387_)) * select(0f, 1f, (_e8403_ <= 1f))), 1f, (_e8531_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8387_)), 0f, (_e8403_ <= -1f))))));
                                                phi_22813_ = type_30_(array<f32, 4>(((_e7174_.member.member[0] * _e8740_) + (_e483_ * select(select(select(_e8165_, ((select(_e8048_, 0f, _e8067_) * _e8165_) * (_e8213_ + (_e8213_ * (1f - (_e8176_ * _e8213_))))), (select((select(0f, 1f, (-1f <= _e7928_)) * select(0f, 1f, (_e7947_ <= 1f))), 1f, (_e8084_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e7928_)), 0f, (_e7947_ <= -1f)))), ((_e7174_.member.member[1] * _e8741_) + (_e503_ * select(select(select(_e8167_, ((select(_e8049_, 0f, _e8072_) * _e8167_) * (_e8215_ + (_e8215_ * (1f - (_e8181_ * _e8215_))))), (select((select(0f, 1f, (-1f <= _e7933_)) * select(0f, 1f, (_e7951_ <= 1f))), 1f, (_e8085_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e7933_)), 0f, (_e7951_ <= -1f)))), ((_e7174_.member.member[2] * _e8742_) + (_e523_ * select(select(select(_e8169_, ((select(_e8050_, 0f, _e8077_) * _e8169_) * (_e8217_ + (_e8217_ * (1f - (_e8186_ * _e8217_))))), (select((select(0f, 1f, (-1f <= _e7938_)) * select(0f, 1f, (_e7955_ <= 1f))), 1f, (_e8086_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e7938_)), 0f, (_e7955_ <= -1f)))), ((_e7174_.member.member[3] * _e8743_) + (_e543_ * select(select(select(_e8171_, ((select(_e8051_, 0f, _e8082_) * _e8171_) * (_e8219_ + (_e8219_ * (1f - (_e8191_ * _e8219_))))), (select((select(0f, 1f, (-1f <= _e7943_)) * select(0f, 1f, (_e7959_ <= 1f))), 1f, (_e8087_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e7943_)), 0f, (_e7959_ <= -1f))))));
                                            }
                                            let _e8789_ = phi_22812_;
                                            let _e8791_ = phi_22813_;
                                            let _e8852_ = (_e1500_ * 6u);
                                            let _e8853_ = local_9_;
                                            let _e8858_ = bitcast<u32>(_e8853_[0]);
                                            let _e8859_ = (_e8858_ & 2139095040u);
                                            if (_e8859_ == 2139095040u) {
                                                phi_22969_ = true;
                                            } else {
                                                phi_22969_ = (_e8859_ == 0u);
                                            }
                                            let _e8863_ = phi_22969_;
                                            let _e8866_ = bitcast<u32>(_e8853_[1]);
                                            let _e8867_ = (_e8866_ & 2139095040u);
                                            if (_e8867_ == 2139095040u) {
                                                phi_22985_ = true;
                                            } else {
                                                phi_22985_ = (_e8867_ == 0u);
                                            }
                                            let _e8871_ = phi_22985_;
                                            let _e8874_ = bitcast<u32>(_e8853_[2]);
                                            let _e8875_ = (_e8874_ & 2139095040u);
                                            if (_e8875_ == 2139095040u) {
                                                phi_23001_ = true;
                                            } else {
                                                phi_23001_ = (_e8875_ == 0u);
                                            }
                                            let _e8879_ = phi_23001_;
                                            let _e8882_ = bitcast<u32>(_e8853_[3]);
                                            let _e8883_ = (_e8882_ & 2139095040u);
                                            if (_e8883_ == 2139095040u) {
                                                phi_23017_ = true;
                                            } else {
                                                phi_23017_ = (_e8883_ == 0u);
                                            }
                                            let _e8887_ = phi_23017_;
                                            let _e8891_ = (_e8852_ < _e156_);
                                            if _e8891_ {
                                                if _e8891_ {
                                                } else {
                                                    phi_37452_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e8852_] = vec4<f32>(bitcast<f32>(select(_e8858_, 0u, _e8863_)), bitcast<f32>(select(_e8866_, 0u, _e8871_)), bitcast<f32>(select(_e8874_, 0u, _e8879_)), bitcast<f32>(select(_e8882_, 0u, _e8887_)));
                                            }
                                            let _e8894_ = (_e8852_ + 1u);
                                            let _e8895_ = local_10_;
                                            let _e8900_ = bitcast<u32>(_e8895_[0]);
                                            let _e8901_ = (_e8900_ & 2139095040u);
                                            if (_e8901_ == 2139095040u) {
                                                phi_23055_ = true;
                                            } else {
                                                phi_23055_ = (_e8901_ == 0u);
                                            }
                                            let _e8905_ = phi_23055_;
                                            let _e8908_ = bitcast<u32>(_e8895_[1]);
                                            let _e8909_ = (_e8908_ & 2139095040u);
                                            if (_e8909_ == 2139095040u) {
                                                phi_23071_ = true;
                                            } else {
                                                phi_23071_ = (_e8909_ == 0u);
                                            }
                                            let _e8913_ = phi_23071_;
                                            let _e8916_ = bitcast<u32>(_e8895_[2]);
                                            let _e8917_ = (_e8916_ & 2139095040u);
                                            if (_e8917_ == 2139095040u) {
                                                phi_23087_ = true;
                                            } else {
                                                phi_23087_ = (_e8917_ == 0u);
                                            }
                                            let _e8921_ = phi_23087_;
                                            let _e8924_ = bitcast<u32>(_e8895_[3]);
                                            let _e8925_ = (_e8924_ & 2139095040u);
                                            if (_e8925_ == 2139095040u) {
                                                phi_23103_ = true;
                                            } else {
                                                phi_23103_ = (_e8925_ == 0u);
                                            }
                                            let _e8929_ = phi_23103_;
                                            let _e8933_ = (_e8894_ < _e156_);
                                            if _e8933_ {
                                                if _e8933_ {
                                                } else {
                                                    phi_37452_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e8894_] = vec4<f32>(bitcast<f32>(select(_e8900_, 0u, _e8905_)), bitcast<f32>(select(_e8908_, 0u, _e8913_)), bitcast<f32>(select(_e8916_, 0u, _e8921_)), bitcast<f32>(select(_e8924_, 0u, _e8929_)));
                                            }
                                            let _e8936_ = (_e8852_ + 2u);
                                            let _e8941_ = bitcast<u32>(_e8791_.member[0]);
                                            let _e8942_ = (_e8941_ & 2139095040u);
                                            if (_e8942_ == 2139095040u) {
                                                phi_23135_ = true;
                                            } else {
                                                phi_23135_ = (_e8942_ == 0u);
                                            }
                                            let _e8946_ = phi_23135_;
                                            let _e8949_ = bitcast<u32>(_e8791_.member[1]);
                                            let _e8950_ = (_e8949_ & 2139095040u);
                                            if (_e8950_ == 2139095040u) {
                                                phi_23151_ = true;
                                            } else {
                                                phi_23151_ = (_e8950_ == 0u);
                                            }
                                            let _e8954_ = phi_23151_;
                                            let _e8957_ = bitcast<u32>(_e8791_.member[2]);
                                            let _e8958_ = (_e8957_ & 2139095040u);
                                            if (_e8958_ == 2139095040u) {
                                                phi_23167_ = true;
                                            } else {
                                                phi_23167_ = (_e8958_ == 0u);
                                            }
                                            let _e8962_ = phi_23167_;
                                            let _e8965_ = bitcast<u32>(_e8791_.member[3]);
                                            let _e8966_ = (_e8965_ & 2139095040u);
                                            if (_e8966_ == 2139095040u) {
                                                phi_23183_ = true;
                                            } else {
                                                phi_23183_ = (_e8966_ == 0u);
                                            }
                                            let _e8970_ = phi_23183_;
                                            let _e8974_ = (_e8936_ < _e156_);
                                            if _e8974_ {
                                                if _e8974_ {
                                                } else {
                                                    phi_37452_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e8936_] = vec4<f32>(bitcast<f32>(select(_e8941_, 0u, _e8946_)), bitcast<f32>(select(_e8949_, 0u, _e8954_)), bitcast<f32>(select(_e8957_, 0u, _e8962_)), bitcast<f32>(select(_e8965_, 0u, _e8970_)));
                                            }
                                            let _e8977_ = (_e8852_ + 3u);
                                            let _e8982_ = bitcast<u32>(_e8789_.member[0]);
                                            let _e8983_ = (_e8982_ & 2139095040u);
                                            if (_e8983_ == 2139095040u) {
                                                phi_23215_ = true;
                                            } else {
                                                phi_23215_ = (_e8983_ == 0u);
                                            }
                                            let _e8987_ = phi_23215_;
                                            let _e8990_ = bitcast<u32>(_e8789_.member[1]);
                                            let _e8991_ = (_e8990_ & 2139095040u);
                                            if (_e8991_ == 2139095040u) {
                                                phi_23231_ = true;
                                            } else {
                                                phi_23231_ = (_e8991_ == 0u);
                                            }
                                            let _e8995_ = phi_23231_;
                                            let _e8998_ = bitcast<u32>(_e8789_.member[2]);
                                            let _e8999_ = (_e8998_ & 2139095040u);
                                            if (_e8999_ == 2139095040u) {
                                                phi_23247_ = true;
                                            } else {
                                                phi_23247_ = (_e8999_ == 0u);
                                            }
                                            let _e9003_ = phi_23247_;
                                            let _e9006_ = bitcast<u32>(_e8789_.member[3]);
                                            let _e9007_ = (_e9006_ & 2139095040u);
                                            if (_e9007_ == 2139095040u) {
                                                phi_23263_ = true;
                                            } else {
                                                phi_23263_ = (_e9007_ == 0u);
                                            }
                                            let _e9011_ = phi_23263_;
                                            let _e9015_ = (_e8977_ < _e156_);
                                            if _e9015_ {
                                                if _e9015_ {
                                                } else {
                                                    phi_37452_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e8977_] = vec4<f32>(bitcast<f32>(select(_e8982_, 0u, _e8987_)), bitcast<f32>(select(_e8990_, 0u, _e8995_)), bitcast<f32>(select(_e8998_, 0u, _e9003_)), bitcast<f32>(select(_e9006_, 0u, _e9011_)));
                                            }
                                            let _e9018_ = (_e8852_ + 4u);
                                            let _e9019_ = bitcast<u32>((_e7174_.member_1_.member[0] + (_e454_.x * (_e8791_.member[0] * _e8791_.member[1]))));
                                            let _e9020_ = (_e9019_ & 2139095040u);
                                            if (_e9020_ == 2139095040u) {
                                                phi_23295_ = true;
                                            } else {
                                                phi_23295_ = (_e9020_ == 0u);
                                            }
                                            let _e9024_ = phi_23295_;
                                            let _e9027_ = bitcast<u32>((_e7174_.member_1_.member[1] + (_e454_.y * (_e8791_.member[1] * _e8791_.member[2]))));
                                            let _e9028_ = (_e9027_ & 2139095040u);
                                            if (_e9028_ == 2139095040u) {
                                                phi_23311_ = true;
                                            } else {
                                                phi_23311_ = (_e9028_ == 0u);
                                            }
                                            let _e9032_ = phi_23311_;
                                            let _e9035_ = bitcast<u32>((_e7174_.member_1_.member[2] + (_e454_.z * (_e8791_.member[2] * _e8791_.member[3]))));
                                            let _e9036_ = (_e9035_ & 2139095040u);
                                            if (_e9036_ == 2139095040u) {
                                                phi_23327_ = true;
                                            } else {
                                                phi_23327_ = (_e9036_ == 0u);
                                            }
                                            let _e9040_ = phi_23327_;
                                            let _e9043_ = bitcast<u32>((_e7174_.member_1_.member[3] + (_e454_.w * (_e8791_.member[3] * _e8791_.member[0]))));
                                            let _e9044_ = (_e9043_ & 2139095040u);
                                            if (_e9044_ == 2139095040u) {
                                                phi_23343_ = true;
                                            } else {
                                                phi_23343_ = (_e9044_ == 0u);
                                            }
                                            let _e9048_ = phi_23343_;
                                            let _e9052_ = (_e9018_ < _e156_);
                                            if _e9052_ {
                                                if _e9052_ {
                                                } else {
                                                    phi_37452_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9018_] = vec4<f32>(bitcast<f32>(select(_e9019_, 0u, _e9024_)), bitcast<f32>(select(_e9027_, 0u, _e9032_)), bitcast<f32>(select(_e9035_, 0u, _e9040_)), bitcast<f32>(select(_e9043_, 0u, _e9048_)));
                                            }
                                            let _e9055_ = (_e8852_ + 5u);
                                            let _e9056_ = bitcast<u32>((_e7834_.member_1_.member[0] + (_e454_.x * (_e8789_.member[0] * _e8789_.member[1]))));
                                            let _e9057_ = (_e9056_ & 2139095040u);
                                            if (_e9057_ == 2139095040u) {
                                                phi_23375_ = true;
                                            } else {
                                                phi_23375_ = (_e9057_ == 0u);
                                            }
                                            let _e9061_ = phi_23375_;
                                            let _e9064_ = bitcast<u32>((_e7834_.member_1_.member[1] + (_e454_.y * (_e8789_.member[1] * _e8789_.member[2]))));
                                            let _e9065_ = (_e9064_ & 2139095040u);
                                            if (_e9065_ == 2139095040u) {
                                                phi_23391_ = true;
                                            } else {
                                                phi_23391_ = (_e9065_ == 0u);
                                            }
                                            let _e9069_ = phi_23391_;
                                            let _e9072_ = bitcast<u32>((_e7834_.member_1_.member[2] + (_e454_.z * (_e8789_.member[2] * _e8789_.member[3]))));
                                            let _e9073_ = (_e9072_ & 2139095040u);
                                            if (_e9073_ == 2139095040u) {
                                                phi_23407_ = true;
                                            } else {
                                                phi_23407_ = (_e9073_ == 0u);
                                            }
                                            let _e9077_ = phi_23407_;
                                            let _e9080_ = bitcast<u32>((_e7834_.member_1_.member[3] + (_e454_.w * (_e8789_.member[3] * _e8789_.member[0]))));
                                            let _e9081_ = (_e9080_ & 2139095040u);
                                            if (_e9081_ == 2139095040u) {
                                                phi_23423_ = true;
                                            } else {
                                                phi_23423_ = (_e9081_ == 0u);
                                            }
                                            let _e9085_ = phi_23423_;
                                            let _e9089_ = (_e9055_ < _e156_);
                                            if _e9089_ {
                                                if _e9089_ {
                                                } else {
                                                    phi_37452_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9055_] = vec4<f32>(bitcast<f32>(select(_e9056_, 0u, _e9061_)), bitcast<f32>(select(_e9064_, 0u, _e9069_)), bitcast<f32>(select(_e9072_, 0u, _e9077_)), bitcast<f32>(select(_e9080_, 0u, _e9085_)));
                                            }
                                            if ((_e158_ + 1u) == _e1495_) {
                                                phi_23447_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e9095_ = phi_23447_;
                                                    if (_e9095_.member < _e9095_.member_1_) {
                                                        phi_23448_ = type_15_((_e9095_.member + 1u), _e9095_.member_1_);
                                                        phi_23471_ = type_15_(1u, _e9095_.member);
                                                    } else {
                                                        phi_23448_ = _e9095_;
                                                        phi_23471_ = type_15_(0u, u32());
                                                    }
                                                    let _e9108_ = phi_23448_;
                                                    let _e9110_ = phi_23471_;
                                                    let _e9114_ = (bitcast<i32>(_e9110_.member) != 0i);
                                                    if _e9114_ {
                                                        let _e9115_ = (_e9110_.member_1_ < 4u);
                                                        if _e9115_ {
                                                        } else {
                                                            phi_36618_ = true;
                                                            break;
                                                        }
                                                        let _e9117_ = local_11_[_e9110_.member_1_];
                                                        let _e9118_ = bitcast<u32>(_e9117_);
                                                        let _e9119_ = (_e9118_ & 2139095040u);
                                                        if (_e9119_ == 2139095040u) {
                                                            phi_23496_ = true;
                                                        } else {
                                                            phi_23496_ = (_e9119_ == 0u);
                                                        }
                                                        let _e9123_ = phi_23496_;
                                                        let _e9125_ = bitcast<f32>(select(_e9118_, 0u, _e9123_));
                                                        if (_e9125_ < -1000000000000000000f) {
                                                            phi_23513_ = -1000000000000000000f;
                                                        } else {
                                                            phi_23513_ = select(_e9125_, 1000000000000000000f, (_e9125_ > 1000000000000000000f));
                                                        }
                                                        let _e9130_ = phi_23513_;
                                                        let _e9131_ = (24u + _e9110_.member_1_);
                                                        if (_e9131_ < 256u) {
                                                        } else {
                                                            phi_36618_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e9131_] = _e9130_;
                                                        if _e9115_ {
                                                        } else {
                                                            phi_36618_ = true;
                                                            break;
                                                        }
                                                        let _e9135_ = local_12_[_e9110_.member_1_];
                                                        let _e9136_ = bitcast<u32>(_e9135_);
                                                        let _e9137_ = (_e9136_ & 2139095040u);
                                                        if (_e9137_ == 2139095040u) {
                                                            phi_23538_ = true;
                                                        } else {
                                                            phi_23538_ = (_e9137_ == 0u);
                                                        }
                                                        let _e9141_ = phi_23538_;
                                                        let _e9143_ = bitcast<f32>(select(_e9136_, 0u, _e9141_));
                                                        if (_e9143_ < -1000000000000000000f) {
                                                            phi_23555_ = -1000000000000000000f;
                                                        } else {
                                                            phi_23555_ = select(_e9143_, 1000000000000000000f, (_e9143_ > 1000000000000000000f));
                                                        }
                                                        let _e9148_ = phi_23555_;
                                                        let _e9149_ = (28u + _e9110_.member_1_);
                                                        if (_e9149_ < 256u) {
                                                        } else {
                                                            phi_36618_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e9149_] = _e9148_;
                                                    }
                                                    phi_23447_ = _e9108_;
                                                    phi_36618_ = _e6495_;
                                                    if !(_e9114_) {
                                                        break;
                                                    }
                                                }
                                                let _e9154_ = phi_36618_;
                                                phi_37452_ = _e9154_;
                                                if _e9154_ {
                                                    break;
                                                }
                                                phi_37457_ = _e9154_;
                                            } else {
                                                phi_37457_ = _e6495_;
                                            }
                                            let _e9156_ = phi_37457_;
                                            phi_37456_ = _e9156_;
                                        } else {
                                            phi_37456_ = _e6418_;
                                        }
                                        let _e9158_ = phi_37456_;
                                        phi_37455_ = _e9158_;
                                    } else {
                                        phi_37455_ = _e6418_;
                                    }
                                    let _e9160_ = phi_37455_;
                                    workgroupBarrier();
                                    phi_37454_ = _e9160_;
                                } else {
                                    phi_37454_ = _e1466_;
                                }
                                let _e9162_ = phi_37454_;
                                phi_37453_ = _e9162_;
                            } else {
                                phi_37453_ = _e1466_;
                            }
                            let _e9164_ = phi_37453_;
                            phi_34888_ = _e9164_;
                            phi_2685_ = _e1481_;
                            phi_37452_ = _e9164_;
                            if !(_e1487_) {
                                break;
                            }
                        }
                        let _e9167_ = phi_37452_;
                        phi_37458_ = _e9167_;
                        if _e9167_ {
                            break;
                        }
                        phi_37459_ = _e9167_;
                    } else {
                        phi_37459_ = _e1075_;
                    }
                    let _e9169_ = phi_37459_;
                    phi_34891_ = _e9169_;
                    phi_1960_ = _e1090_;
                    phi_37458_ = _e9169_;
                    if !(_e1096_) {
                        break;
                    }
                }
                let _e9172_ = phi_37458_;
                if _e9172_ {
                    break;
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn fused_core_scan_256(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    wg_id_1_ = wg_id;
    num_wg_1_ = num_wg;
    lid_1_ = lid;
    function_();
    return;
}
