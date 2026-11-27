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
    var local_15_: array<f32, 4>;
    var local_16_: array<f32, 4>;
    var local_17_: array<f32, 4>;
    var local_18_: array<f32, 4>;
    var local_19_: array<f32, 4>;
    var local_20_: array<f32, 4>;
    var phi_302_: u32;
    var phi_303_: u32;
    var phi_320_: bool;
    var phi_328_: bool;
    var phi_338_: u32;
    var phi_369_: u32;
    var phi_380_: u32;
    var phi_383_: u32;
    var phi_381_: u32;
    var phi_384_: u32;
    var phi_395_: u32;
    var phi_398_: u32;
    var phi_400_: u32;
    var phi_402_: u32;
    var phi_416_: u32;
    var phi_417_: u32;
    var phi_396_: u32;
    var phi_399_: u32;
    var phi_401_: u32;
    var phi_403_: u32;
    var local_21_: u32;
    var phi_421_: u32;
    var phi_422_: u32;
    var phi_436_: u32;
    var phi_437_: u32;
    var phi_451_: vec4<f32>;
    var phi_464_: vec4<f32>;
    var phi_477_: vec4<f32>;
    var phi_490_: vec4<f32>;
    var phi_503_: vec4<f32>;
    var phi_516_: vec4<f32>;
    var phi_529_: vec4<f32>;
    var phi_542_: vec4<f32>;
    var phi_555_: vec4<f32>;
    var phi_568_: vec4<f32>;
    var phi_581_: vec4<f32>;
    var phi_594_: vec4<f32>;
    var phi_607_: vec4<f32>;
    var phi_620_: vec4<f32>;
    var phi_633_: vec4<f32>;
    var phi_646_: vec4<f32>;
    var phi_659_: vec4<f32>;
    var phi_672_: vec4<f32>;
    var phi_685_: vec4<f32>;
    var phi_698_: vec4<f32>;
    var phi_711_: vec4<f32>;
    var phi_724_: vec4<f32>;
    var phi_737_: vec4<f32>;
    var phi_763_: vec4<f32>;
    var phi_790_: bool;
    var phi_796_: f32;
    var phi_797_: f32;
    var phi_804_: f32;
    var phi_830_: bool;
    var phi_836_: f32;
    var phi_837_: f32;
    var phi_844_: f32;
    var phi_870_: bool;
    var phi_876_: f32;
    var phi_877_: f32;
    var phi_884_: f32;
    var phi_910_: bool;
    var phi_916_: f32;
    var phi_917_: f32;
    var phi_924_: f32;
    var phi_950_: bool;
    var phi_956_: f32;
    var phi_957_: f32;
    var phi_964_: f32;
    var phi_990_: bool;
    var phi_996_: f32;
    var phi_997_: f32;
    var phi_1004_: f32;
    var phi_1030_: bool;
    var phi_1036_: f32;
    var phi_1037_: f32;
    var phi_1044_: f32;
    var phi_1070_: bool;
    var phi_1076_: f32;
    var phi_1077_: f32;
    var phi_1084_: f32;
    var phi_1120_: vec4<f32>;
    var phi_1132_: bool;
    var phi_1149_: bool;
    var phi_1166_: bool;
    var phi_1183_: bool;
    var phi_1207_: vec4<f32>;
    var phi_1219_: bool;
    var phi_1236_: bool;
    var phi_1253_: bool;
    var phi_1270_: bool;
    var phi_1294_: vec4<f32>;
    var phi_1306_: bool;
    var phi_1323_: bool;
    var phi_1340_: bool;
    var phi_1357_: bool;
    var phi_1381_: vec4<f32>;
    var phi_1393_: bool;
    var phi_1410_: bool;
    var phi_1427_: bool;
    var phi_1444_: bool;
    var phi_1468_: vec4<f32>;
    var phi_1480_: bool;
    var phi_1497_: bool;
    var phi_1514_: bool;
    var phi_1531_: bool;
    var phi_1555_: vec4<f32>;
    var phi_1567_: bool;
    var phi_1584_: bool;
    var phi_1601_: bool;
    var phi_1618_: bool;
    var phi_1642_: vec4<f32>;
    var phi_1654_: bool;
    var phi_1671_: f32;
    var phi_1683_: bool;
    var phi_1700_: f32;
    var phi_1712_: bool;
    var phi_1729_: f32;
    var phi_1741_: bool;
    var phi_1758_: f32;
    var phi_1777_: vec4<f32>;
    var phi_1789_: bool;
    var phi_1806_: f32;
    var phi_1818_: bool;
    var phi_1835_: f32;
    var phi_1847_: bool;
    var phi_1864_: f32;
    var phi_1876_: bool;
    var phi_1893_: f32;
    var phi_1902_: type_15_;
    var phi_1903_: type_15_;
    var phi_1926_: type_15_;
    var phi_33640_: bool;
    var phi_33836_: bool;
    var phi_33835_: bool;
    var phi_2046_: bool;
    var phi_33834_: bool;
    var phi_2048_: bool;
    var phi_2052_: bool;
    var phi_2053_: bool;
    var phi_33833_: bool;
    var phi_2054_: bool;
    var phi_2057_: bool;
    var phi_2058_: bool;
    var phi_2059_: bool;
    var phi_2069_: bool;
    var phi_2076_: type_15_;
    var phi_2077_: type_15_;
    var phi_2100_: type_15_;
    var phi_33828_: bool;
    var phi_33869_: bool;
    var phi_33866_: bool;
    var phi_2126_: type_15_;
    var phi_2127_: type_15_;
    var phi_2150_: type_15_;
    var phi_2171_: bool;
    var phi_2252_: bool;
    var phi_2268_: bool;
    var phi_2284_: bool;
    var phi_2300_: bool;
    var phi_2328_: bool;
    var phi_2344_: bool;
    var phi_2360_: bool;
    var phi_2376_: bool;
    var phi_2404_: bool;
    var phi_2420_: bool;
    var phi_2436_: bool;
    var phi_2452_: bool;
    var phi_2480_: bool;
    var phi_2496_: bool;
    var phi_2512_: bool;
    var phi_2528_: bool;
    var phi_2556_: bool;
    var phi_2572_: bool;
    var phi_2588_: bool;
    var phi_2604_: bool;
    var phi_2632_: bool;
    var phi_2648_: bool;
    var phi_2664_: bool;
    var phi_2680_: bool;
    var phi_2708_: bool;
    var phi_2724_: bool;
    var phi_2740_: bool;
    var phi_2756_: bool;
    var phi_2784_: bool;
    var phi_2800_: bool;
    var phi_2816_: bool;
    var phi_2832_: bool;
    var phi_33863_: bool;
    var phi_2851_: type_15_;
    var phi_2852_: type_15_;
    var phi_2875_: type_15_;
    var phi_2890_: u32;
    var phi_2919_: vec4<f32>;
    var phi_2932_: vec4<f32>;
    var phi_2933_: vec4<f32>;
    var phi_2934_: vec4<f32>;
    var phi_33858_: bool;
    var phi_6740_: type_15_;
    var phi_6741_: type_15_;
    var phi_6764_: type_15_;
    var phi_6859_: f32;
    var phi_6870_: f32;
    var phi_6881_: f32;
    var phi_6892_: f32;
    var phi_6903_: f32;
    var phi_6914_: f32;
    var phi_6925_: f32;
    var phi_6936_: f32;
    var phi_6947_: f32;
    var phi_6958_: f32;
    var phi_7266_: f32;
    var phi_7277_: bool;
    var phi_7283_: f32;
    var phi_7288_: u32;
    var phi_7294_: u32;
    var phi_7327_: f32;
    var phi_7332_: f32;
    var phi_7349_: f32;
    var phi_7350_: f32;
    var phi_7351_: f32;
    var phi_7352_: f32;
    var phi_7369_: f32;
    var phi_7372_: f32;
    var phi_7374_: f32;
    var phi_7376_: f32;
    var phi_7378_: u32;
    var phi_7380_: f32;
    var phi_7382_: f32;
    var phi_7421_: f32;
    var phi_7422_: f32;
    var phi_7370_: f32;
    var phi_7373_: f32;
    var phi_7375_: f32;
    var phi_7377_: f32;
    var phi_7379_: u32;
    var phi_7381_: f32;
    var phi_7383_: f32;
    var phi_33837_: bool;
    var phi_7474_: f32;
    var phi_7475_: f32;
    var phi_7485_: u32;
    var phi_7504_: array<f32, 2>;
    var phi_7759_: bool;
    var phi_7782_: bool;
    var phi_7810_: bool;
    var phi_7834_: bool;
    var phi_33954_: bool;
    var phi_33953_: bool;
    var phi_5744_: type_15_;
    var phi_5745_: type_15_;
    var phi_5768_: type_15_;
    var phi_5786_: f32;
    var phi_5797_: bool;
    var phi_5822_: f32;
    var phi_5833_: bool;
    var phi_33970_: bool;
    var phi_33997_: bool;
    var phi_5961_: f32;
    var phi_5978_: f32;
    var phi_5989_: f32;
    var phi_6000_: f32;
    var phi_6011_: f32;
    var phi_6022_: f32;
    var phi_6237_: f32;
    var phi_6312_: f32;
    var phi_6317_: u32;
    var phi_6323_: u32;
    var phi_6348_: type_15_;
    var phi_6351_: f32;
    var phi_6353_: f32;
    var phi_6349_: type_15_;
    var phi_6376_: type_15_;
    var phi_23420_: bool;
    var phi_23480_: bool;
    var phi_23540_: bool;
    var phi_23600_: bool;
    var phi_6543_: u32;
    var phi_6544_: u32;
    var phi_6595_: u32;
    var phi_6596_: u32;
    var phi_6352_: f32;
    var phi_6354_: f32;
    var phi_33982_: bool;
    var local_22_: f32;
    var phi_6704_: bool;
    var local_23_: f32;
    var phi_6726_: bool;
    var phi_34292_: bool;
    var phi_34060_: bool;
    var phi_2943_: type_15_;
    var phi_2944_: type_15_;
    var phi_2967_: type_15_;
    var phi_3080_: f32;
    var phi_3097_: f32;
    var phi_3108_: f32;
    var phi_3119_: f32;
    var phi_3130_: f32;
    var phi_3141_: f32;
    var phi_3356_: f32;
    var phi_3431_: f32;
    var phi_3449_: f32;
    var phi_3460_: bool;
    var phi_3466_: f32;
    var phi_3475_: u32;
    var phi_34048_: bool;
    var phi_3494_: type_15_;
    var phi_3497_: array<f32, 2>;
    var phi_3499_: f32;
    var phi_3501_: f32;
    var phi_3503_: array<f32, 2>;
    var phi_3505_: bool;
    var phi_3495_: type_15_;
    var phi_3528_: type_15_;
    var phi_23660_: bool;
    var phi_23675_: bool;
    var phi_23690_: bool;
    var phi_23705_: bool;
    var phi_3576_: type_31_;
    var phi_3639_: u32;
    var phi_3642_: type_31_;
    var phi_3744_: type_31_;
    var phi_3640_: u32;
    var phi_3643_: type_31_;
    var phi_34034_: bool;
    var phi_3868_: array<f32, 2>;
    var phi_3880_: u32;
    var phi_3944_: array<f32, 2>;
    var phi_23720_: bool;
    var phi_4059_: bool;
    var phi_23780_: bool;
    var phi_4024_: bool;
    var phi_4065_: f32;
    var phi_4066_: f32;
    var phi_4067_: f32;
    var phi_4068_: bool;
    var phi_4074_: f32;
    var phi_4076_: f32;
    var phi_4077_: bool;
    var phi_23840_: bool;
    var phi_4193_: bool;
    var phi_23900_: bool;
    var phi_4158_: bool;
    var phi_4199_: f32;
    var phi_4200_: f32;
    var phi_4201_: f32;
    var phi_4202_: bool;
    var phi_4208_: f32;
    var phi_4210_: f32;
    var phi_4211_: bool;
    var phi_23960_: bool;
    var phi_4343_: bool;
    var phi_24020_: bool;
    var phi_4308_: bool;
    var phi_4349_: f32;
    var phi_4350_: f32;
    var phi_4351_: f32;
    var phi_4352_: bool;
    var phi_4358_: f32;
    var phi_4360_: f32;
    var phi_4361_: bool;
    var phi_24080_: bool;
    var phi_4479_: bool;
    var phi_24140_: bool;
    var phi_4444_: bool;
    var phi_4485_: f32;
    var phi_4486_: f32;
    var phi_4487_: f32;
    var phi_4488_: bool;
    var phi_4494_: f32;
    var phi_4496_: f32;
    var phi_4497_: bool;
    var phi_4512_: f32;
    var phi_4514_: f32;
    var phi_4515_: bool;
    var phi_4520_: f32;
    var phi_4558_: u32;
    var phi_4559_: u32;
    var phi_34066_: bool;
    var phi_4619_: bool;
    var phi_4624_: bool;
    var phi_34159_: bool;
    var phi_4625_: array<f32, 2>;
    var phi_4626_: f32;
    var phi_4627_: f32;
    var phi_4628_: array<f32, 2>;
    var phi_4629_: bool;
    var phi_34158_: bool;
    var phi_3498_: array<f32, 2>;
    var phi_3500_: f32;
    var phi_3502_: f32;
    var phi_3504_: array<f32, 2>;
    var phi_3506_: bool;
    var phi_34157_: bool;
    var local_24_: bool;
    var phi_4992_: type_15_;
    var phi_4995_: f32;
    var phi_4997_: f32;
    var phi_4993_: type_15_;
    var phi_5020_: type_15_;
    var phi_24200_: bool;
    var phi_24260_: bool;
    var phi_24320_: bool;
    var phi_24380_: bool;
    var phi_5186_: u32;
    var phi_5187_: u32;
    var phi_5238_: u32;
    var phi_5239_: u32;
    var phi_4996_: f32;
    var phi_4998_: f32;
    var phi_34163_: bool;
    var local_25_: f32;
    var local_26_: f32;
    var phi_34210_: bool;
    var phi_5371_: type_15_;
    var phi_5374_: f32;
    var phi_5376_: f32;
    var phi_5372_: type_15_;
    var phi_5399_: type_15_;
    var phi_24440_: bool;
    var phi_24500_: bool;
    var phi_24560_: bool;
    var phi_24620_: bool;
    var phi_5566_: u32;
    var phi_5567_: u32;
    var phi_5618_: u32;
    var phi_5619_: u32;
    var phi_5375_: f32;
    var phi_5377_: f32;
    var phi_34206_: bool;
    var local_27_: f32;
    var local_28_: f32;
    var phi_34250_: bool;
    var phi_24680_: bool;
    var phi_24695_: bool;
    var phi_24710_: bool;
    var local_29_: f32;
    var local_30_: f32;
    var phi_24725_: bool;
    var local_31_: f32;
    var local_32_: f32;
    var local_33_: f32;
    var local_34_: f32;
    var local_35_: array<f32, 2>;
    var local_36_: array<f32, 2>;
    var local_37_: array<f32, 2>;
    var local_38_: array<f32, 2>;
    var phi_34249_: bool;
    var phi_34248_: bool;
    var phi_34247_: bool;
    var phi_34291_: bool;
    var phi_34290_: bool;
    var phi_34289_: bool;
    var phi_7849_: type_15_;
    var phi_7850_: type_15_;
    var phi_7873_: type_15_;
    var phi_7891_: f32;
    var phi_7902_: f32;
    var phi_7913_: f32;
    var phi_7924_: f32;
    var phi_7935_: f32;
    var phi_8042_: bool;
    var phi_8048_: f32;
    var phi_8049_: f32;
    var phi_8056_: f32;
    var phi_8081_: bool;
    var phi_8087_: f32;
    var phi_8088_: f32;
    var phi_8095_: f32;
    var phi_8120_: bool;
    var phi_8126_: f32;
    var phi_8127_: f32;
    var phi_8134_: f32;
    var phi_8159_: bool;
    var phi_8165_: f32;
    var phi_8166_: f32;
    var phi_8173_: f32;
    var phi_8198_: bool;
    var phi_8204_: f32;
    var phi_8205_: f32;
    var phi_8212_: f32;
    var phi_8307_: f32;
    var phi_8318_: f32;
    var phi_8388_: bool;
    var phi_8435_: bool;
    var phi_8452_: bool;
    var phi_8464_: f32;
    var phi_8476_: f32;
    var phi_8487_: bool;
    var phi_24740_: bool;
    var phi_24800_: bool;
    var phi_24860_: bool;
    var phi_24920_: bool;
    var phi_8977_: f32;
    var phi_8990_: f32;
    var phi_9004_: bool;
    var phi_9021_: bool;
    var phi_9029_: type_28_;
    var phi_9070_: u32;
    var phi_9073_: type_28_;
    var phi_9113_: type_28_;
    var phi_9071_: u32;
    var phi_9074_: type_28_;
    var phi_34263_: bool;
    var local_39_: type_28_;
    var local_40_: type_28_;
    var phi_9149_: u32;
    var phi_9170_: f32;
    var phi_9184_: bool;
    var phi_9215_: bool;
    var phi_9242_: bool;
    var phi_34416_: bool;
    var phi_34415_: bool;
    var phi_34414_: bool;
    var phi_16635_: f32;
    var phi_16646_: f32;
    var phi_16657_: f32;
    var phi_16668_: f32;
    var phi_16679_: f32;
    var phi_16786_: bool;
    var phi_16792_: f32;
    var phi_16793_: f32;
    var phi_16800_: f32;
    var phi_16825_: bool;
    var phi_16831_: f32;
    var phi_16832_: f32;
    var phi_16839_: f32;
    var phi_16864_: bool;
    var phi_16870_: f32;
    var phi_16871_: f32;
    var phi_16878_: f32;
    var phi_16903_: bool;
    var phi_16909_: f32;
    var phi_16910_: f32;
    var phi_16917_: f32;
    var phi_16942_: bool;
    var phi_16948_: f32;
    var phi_16949_: f32;
    var phi_16956_: f32;
    var phi_17066_: f32;
    var phi_17077_: f32;
    var phi_17147_: bool;
    var phi_17173_: u32;
    var phi_17183_: type_15_;
    var phi_17186_: f32;
    var phi_17184_: type_15_;
    var phi_17209_: type_15_;
    var phi_17255_: u32;
    var phi_17256_: u32;
    var phi_25100_: bool;
    var phi_25160_: bool;
    var phi_25220_: bool;
    var phi_25280_: bool;
    var phi_25340_: bool;
    var phi_25400_: bool;
    var phi_25460_: bool;
    var phi_25520_: bool;
    var phi_17997_: u32;
    var phi_17998_: u32;
    var phi_17187_: f32;
    var phi_34428_: bool;
    var local_41_: f32;
    var phi_18052_: bool;
    var phi_35384_: bool;
    var phi_35383_: bool;
    var phi_34610_: bool;
    var phi_9260_: type_15_;
    var phi_9261_: type_15_;
    var phi_9284_: type_15_;
    var phi_9302_: f32;
    var phi_9313_: f32;
    var phi_9324_: f32;
    var phi_9335_: f32;
    var phi_9346_: f32;
    var phi_9453_: bool;
    var phi_9459_: f32;
    var phi_9460_: f32;
    var phi_9467_: f32;
    var phi_9492_: bool;
    var phi_9498_: f32;
    var phi_9499_: f32;
    var phi_9506_: f32;
    var phi_9531_: bool;
    var phi_9537_: f32;
    var phi_9538_: f32;
    var phi_9545_: f32;
    var phi_9570_: bool;
    var phi_9576_: f32;
    var phi_9577_: f32;
    var phi_9584_: f32;
    var phi_9609_: bool;
    var phi_9615_: f32;
    var phi_9616_: f32;
    var phi_9623_: f32;
    var phi_9733_: f32;
    var phi_9744_: f32;
    var phi_9814_: bool;
    var phi_9841_: f32;
    var phi_9861_: f32;
    var phi_34582_: bool;
    var phi_9892_: type_15_;
    var phi_9895_: f32;
    var phi_9897_: bool;
    var phi_9893_: type_15_;
    var phi_9920_: type_15_;
    var phi_9943_: f32;
    var phi_9972_: u32;
    var phi_9973_: u32;
    var phi_25700_: bool;
    var phi_25760_: bool;
    var phi_25820_: bool;
    var phi_25880_: bool;
    var phi_25940_: bool;
    var phi_26000_: bool;
    var phi_26060_: bool;
    var phi_26120_: bool;
    var phi_10714_: u32;
    var phi_10715_: u32;
    var phi_10759_: bool;
    var phi_10775_: bool;
    var phi_10791_: bool;
    var phi_10808_: bool;
    var phi_10820_: f32;
    var phi_10832_: f32;
    var phi_26300_: bool;
    var phi_26360_: bool;
    var phi_26480_: bool;
    var phi_26540_: bool;
    var phi_26600_: bool;
    var phi_11328_: f32;
    var phi_11336_: f32;
    var phi_11465_: bool;
    var phi_11471_: f32;
    var phi_11472_: f32;
    var phi_11391_: f32;
    var phi_11393_: f32;
    var phi_11394_: bool;
    var phi_11414_: f32;
    var phi_11473_: f32;
    var phi_11476_: type_28_;
    var phi_26660_: bool;
    var phi_11593_: bool;
    var phi_26720_: bool;
    var phi_11558_: bool;
    var phi_11599_: f32;
    var phi_11600_: f32;
    var phi_11601_: f32;
    var phi_11602_: bool;
    var phi_11608_: f32;
    var phi_11610_: f32;
    var phi_11611_: bool;
    var phi_11616_: f32;
    var phi_11654_: u32;
    var phi_11655_: u32;
    var phi_34573_: bool;
    var phi_11714_: type_28_;
    var phi_11742_: u32;
    var phi_11745_: type_28_;
    var phi_11785_: type_28_;
    var phi_11743_: u32;
    var phi_11746_: type_28_;
    var phi_34864_: bool;
    var local_42_: type_28_;
    var local_43_: type_28_;
    var phi_11821_: u32;
    var phi_34879_: bool;
    var phi_11843_: type_33_;
    var phi_11846_: f32;
    var phi_11848_: f32;
    var phi_11850_: bool;
    var phi_11852_: bool;
    var phi_11879_: type_34_;
    var phi_11844_: type_33_;
    var phi_11893_: type_34_;
    var phi_12025_: bool;
    var phi_12031_: f32;
    var phi_12032_: f32;
    var phi_11951_: f32;
    var phi_11953_: f32;
    var phi_11954_: bool;
    var phi_11974_: f32;
    var phi_12033_: f32;
    var phi_12034_: f32;
    var phi_12055_: f32;
    var phi_12084_: u32;
    var phi_12085_: u32;
    var phi_26780_: bool;
    var phi_26840_: bool;
    var phi_26900_: bool;
    var phi_26960_: bool;
    var phi_27020_: bool;
    var phi_27080_: bool;
    var phi_27140_: bool;
    var phi_27200_: bool;
    var phi_12826_: u32;
    var phi_12827_: u32;
    var phi_27740_: bool;
    var phi_13661_: bool;
    var phi_27800_: bool;
    var phi_13626_: bool;
    var phi_13667_: f32;
    var phi_13668_: f32;
    var phi_13669_: f32;
    var phi_13670_: bool;
    var phi_13676_: f32;
    var phi_13678_: f32;
    var phi_13679_: bool;
    var phi_13684_: f32;
    var phi_13722_: u32;
    var phi_13723_: u32;
    var phi_34870_: bool;
    var phi_13779_: bool;
    var phi_13797_: bool;
    var phi_35020_: bool;
    var phi_13799_: f32;
    var phi_13800_: f32;
    var phi_13801_: bool;
    var phi_13802_: bool;
    var phi_35019_: bool;
    var phi_11847_: f32;
    var phi_11849_: f32;
    var phi_11851_: bool;
    var phi_11853_: bool;
    var phi_35018_: bool;
    var phi_35027_: bool;
    var phi_13803_: f32;
    var phi_13804_: bool;
    var phi_35026_: bool;
    var phi_13805_: f32;
    var phi_13806_: bool;
    var phi_35025_: bool;
    var phi_9896_: f32;
    var phi_9898_: bool;
    var phi_35024_: bool;
    var phi_13820_: f32;
    var local_44_: f32;
    var phi_13849_: u32;
    var phi_13850_: u32;
    var phi_27860_: bool;
    var phi_27920_: bool;
    var phi_27980_: bool;
    var phi_28040_: bool;
    var phi_28100_: bool;
    var phi_28160_: bool;
    var phi_28220_: bool;
    var phi_28280_: bool;
    var phi_14591_: u32;
    var phi_14592_: u32;
    var phi_14636_: bool;
    var phi_14652_: bool;
    var phi_14668_: bool;
    var phi_14685_: bool;
    var phi_14697_: f32;
    var phi_14709_: f32;
    var phi_28460_: bool;
    var phi_28520_: bool;
    var phi_28640_: bool;
    var phi_28700_: bool;
    var phi_28760_: bool;
    var phi_15205_: f32;
    var phi_15213_: f32;
    var phi_15342_: bool;
    var phi_15348_: f32;
    var phi_15349_: f32;
    var phi_15268_: f32;
    var phi_15270_: f32;
    var phi_15271_: bool;
    var phi_15291_: f32;
    var phi_15350_: f32;
    var phi_15353_: type_28_;
    var local_45_: f32;
    var local_46_: f32;
    var local_47_: f32;
    var phi_28820_: bool;
    var phi_15469_: bool;
    var phi_28880_: bool;
    var phi_15434_: bool;
    var phi_15475_: f32;
    var phi_15476_: f32;
    var phi_15477_: f32;
    var phi_15478_: bool;
    var phi_15484_: f32;
    var phi_15486_: f32;
    var phi_15487_: bool;
    var phi_15492_: f32;
    var phi_15530_: u32;
    var phi_15531_: u32;
    var phi_35028_: bool;
    var phi_16574_: bool;
    var phi_16605_: bool;
    var phi_15610_: u32;
    var phi_15620_: type_15_;
    var phi_15623_: f32;
    var phi_15621_: type_15_;
    var phi_15646_: type_15_;
    var phi_15692_: u32;
    var phi_15693_: u32;
    var phi_28940_: bool;
    var phi_29000_: bool;
    var phi_29060_: bool;
    var phi_29120_: bool;
    var phi_29180_: bool;
    var phi_29240_: bool;
    var phi_29300_: bool;
    var phi_29360_: bool;
    var phi_16434_: u32;
    var phi_16435_: u32;
    var phi_15624_: f32;
    var phi_35263_: bool;
    var local_48_: f32;
    var phi_16489_: bool;
    var phi_35375_: bool;
    var phi_35366_: bool;
    var phi_35365_: bool;
    var phi_35364_: bool;
    var phi_35363_: bool;
    var phi_35382_: bool;
    var phi_18073_: type_15_;
    var phi_18074_: type_15_;
    var phi_18097_: type_15_;
    var phi_35378_: bool;
    var phi_18269_: type_15_;
    var phi_18270_: type_15_;
    var phi_18293_: type_15_;
    var phi_18664_: f32;
    var phi_18675_: f32;
    var phi_18686_: f32;
    var phi_29540_: bool;
    var phi_29555_: bool;
    var phi_18793_: bool;
    var phi_18813_: bool;
    var phi_35387_: bool;
    var phi_35407_: bool;
    var phi_35406_: bool;
    var phi_18924_: type_15_;
    var phi_18925_: type_15_;
    var phi_18948_: type_15_;
    var phi_35402_: bool;
    var phi_18861_: type_15_;
    var phi_18862_: type_15_;
    var phi_18885_: type_15_;
    var phi_35408_: bool;
    var phi_36231_: bool;
    var phi_29570_: bool;
    var phi_29585_: bool;
    var phi_29600_: bool;
    var phi_29615_: bool;
    var phi_29630_: bool;
    var phi_29645_: bool;
    var phi_29660_: bool;
    var phi_29675_: bool;
    var phi_29690_: bool;
    var phi_29705_: bool;
    var phi_29720_: bool;
    var phi_29735_: bool;
    var phi_29750_: bool;
    var phi_29765_: bool;
    var phi_29780_: bool;
    var phi_29795_: bool;
    var phi_29810_: bool;
    var phi_29825_: bool;
    var phi_29840_: bool;
    var phi_29855_: bool;
    var phi_29870_: bool;
    var phi_29885_: bool;
    var phi_29900_: bool;
    var phi_29915_: bool;
    var phi_29930_: bool;
    var phi_29945_: bool;
    var phi_29960_: bool;
    var phi_29975_: bool;
    var phi_29990_: bool;
    var phi_30005_: bool;
    var phi_30020_: bool;
    var phi_30035_: bool;
    var phi_30050_: bool;
    var phi_30065_: bool;
    var phi_30080_: bool;
    var phi_30095_: bool;
    var phi_30110_: bool;
    var phi_30125_: bool;
    var phi_30140_: bool;
    var phi_30155_: bool;
    var phi_30170_: bool;
    var phi_30185_: bool;
    var phi_30200_: bool;
    var phi_30215_: bool;
    var phi_30230_: bool;
    var phi_30245_: bool;
    var phi_30260_: bool;
    var phi_30275_: bool;
    var phi_30290_: bool;
    var phi_30305_: bool;
    var phi_30320_: bool;
    var phi_30335_: bool;
    var phi_30350_: bool;
    var phi_30365_: bool;
    var phi_30380_: bool;
    var phi_30395_: bool;
    var phi_30410_: bool;
    var phi_30425_: bool;
    var phi_30440_: bool;
    var phi_30455_: bool;
    var phi_30470_: bool;
    var phi_30485_: bool;
    var phi_30500_: bool;
    var phi_30515_: bool;
    var phi_19453_: bool;
    var phi_19455_: bool;
    var phi_19456_: bool;
    var phi_19457_: bool;
    var phi_19458_: bool;
    var phi_30530_: bool;
    var phi_30545_: bool;
    var phi_30560_: bool;
    var phi_30575_: bool;
    var phi_19836_: type_37_;
    var phi_30590_: bool;
    var phi_30605_: bool;
    var phi_30620_: bool;
    var phi_30635_: bool;
    var phi_30650_: bool;
    var phi_30665_: bool;
    var phi_30680_: bool;
    var phi_30695_: bool;
    var phi_30710_: bool;
    var phi_30725_: bool;
    var phi_30740_: bool;
    var phi_30755_: bool;
    var phi_30770_: bool;
    var phi_30785_: bool;
    var phi_30800_: bool;
    var phi_30815_: bool;
    var phi_30830_: bool;
    var phi_30845_: bool;
    var phi_30860_: bool;
    var phi_30875_: bool;
    var phi_30890_: bool;
    var phi_30905_: bool;
    var phi_30920_: bool;
    var phi_30935_: bool;
    var phi_30950_: bool;
    var phi_30965_: bool;
    var phi_30980_: bool;
    var phi_30995_: bool;
    var phi_31010_: bool;
    var phi_31025_: bool;
    var phi_31040_: bool;
    var phi_31055_: bool;
    var phi_31070_: bool;
    var phi_31085_: bool;
    var phi_31100_: bool;
    var phi_31115_: bool;
    var phi_31130_: bool;
    var phi_31145_: bool;
    var phi_31160_: bool;
    var phi_31175_: bool;
    var phi_31190_: bool;
    var phi_31205_: bool;
    var phi_31220_: bool;
    var phi_31235_: bool;
    var phi_31250_: bool;
    var phi_31265_: bool;
    var phi_31280_: bool;
    var phi_31295_: bool;
    var phi_31310_: bool;
    var phi_31325_: bool;
    var phi_31340_: bool;
    var phi_31355_: bool;
    var phi_31370_: bool;
    var phi_31385_: bool;
    var phi_31400_: bool;
    var phi_31415_: bool;
    var phi_31430_: bool;
    var phi_31445_: bool;
    var phi_31460_: bool;
    var phi_31475_: bool;
    var phi_31490_: bool;
    var phi_31505_: bool;
    var phi_31520_: bool;
    var phi_31535_: bool;
    var phi_20282_: bool;
    var phi_20284_: bool;
    var phi_20285_: bool;
    var phi_20286_: bool;
    var phi_20287_: bool;
    var phi_31550_: bool;
    var phi_31565_: bool;
    var phi_31580_: bool;
    var phi_31595_: bool;
    var phi_20665_: type_37_;
    var phi_20682_: bool;
    var phi_20684_: bool;
    var phi_20685_: bool;
    var phi_20686_: bool;
    var phi_20687_: bool;
    var phi_31610_: bool;
    var phi_31625_: bool;
    var phi_31640_: bool;
    var phi_31655_: bool;
    var phi_31670_: bool;
    var phi_31685_: bool;
    var phi_31700_: bool;
    var phi_31715_: bool;
    var phi_31730_: bool;
    var phi_31745_: bool;
    var phi_31760_: bool;
    var phi_31775_: bool;
    var phi_31790_: bool;
    var phi_31805_: bool;
    var phi_31820_: bool;
    var phi_31835_: bool;
    var phi_31850_: bool;
    var phi_31865_: bool;
    var phi_31880_: bool;
    var phi_31895_: bool;
    var phi_31910_: bool;
    var phi_31925_: bool;
    var phi_31940_: bool;
    var phi_31955_: bool;
    var phi_31970_: bool;
    var phi_31985_: bool;
    var phi_32000_: bool;
    var phi_32015_: bool;
    var phi_32030_: bool;
    var phi_32045_: bool;
    var phi_32060_: bool;
    var phi_32075_: bool;
    var phi_32090_: bool;
    var phi_32105_: bool;
    var phi_32120_: bool;
    var phi_32135_: bool;
    var phi_32150_: bool;
    var phi_32165_: bool;
    var phi_32180_: bool;
    var phi_32195_: bool;
    var phi_32210_: bool;
    var phi_32225_: bool;
    var phi_32240_: bool;
    var phi_32255_: bool;
    var phi_32510_: bool;
    var phi_32525_: bool;
    var phi_32540_: bool;
    var phi_32555_: bool;
    var phi_32570_: bool;
    var phi_32585_: bool;
    var phi_32600_: bool;
    var phi_32615_: bool;
    var phi_32630_: bool;
    var phi_32645_: bool;
    var phi_32660_: bool;
    var phi_32675_: bool;
    var phi_32690_: bool;
    var phi_32705_: bool;
    var phi_32720_: bool;
    var phi_32735_: bool;
    var phi_32750_: bool;
    var phi_32765_: bool;
    var phi_32780_: bool;
    var phi_32795_: bool;
    var phi_32810_: bool;
    var phi_32825_: bool;
    var phi_32840_: bool;
    var phi_32855_: bool;
    var phi_32870_: bool;
    var phi_32885_: bool;
    var phi_32900_: bool;
    var phi_32915_: bool;
    var phi_32930_: bool;
    var phi_32945_: bool;
    var phi_32960_: bool;
    var phi_32975_: bool;
    var phi_32990_: bool;
    var phi_33005_: bool;
    var phi_33020_: bool;
    var phi_33035_: bool;
    var phi_33050_: bool;
    var phi_33065_: bool;
    var phi_33080_: bool;
    var phi_33095_: bool;
    var phi_33110_: bool;
    var phi_33125_: bool;
    var phi_33140_: bool;
    var phi_33155_: bool;
    var phi_33170_: bool;
    var phi_33185_: bool;
    var phi_33200_: bool;
    var phi_33215_: bool;
    var phi_33230_: bool;
    var phi_33245_: bool;
    var phi_33260_: bool;
    var phi_33275_: bool;
    var phi_33530_: bool;
    var phi_33545_: bool;
    var phi_33560_: bool;
    var phi_33575_: bool;
    var phi_33590_: bool;
    var phi_33605_: bool;
    var phi_33620_: bool;
    var phi_33635_: bool;
    var phi_22595_: type_30_;
    var phi_22596_: type_30_;
    var phi_22752_: bool;
    var phi_22768_: bool;
    var phi_22784_: bool;
    var phi_22800_: bool;
    var phi_22838_: bool;
    var phi_22854_: bool;
    var phi_22870_: bool;
    var phi_22886_: bool;
    var phi_22918_: bool;
    var phi_22934_: bool;
    var phi_22950_: bool;
    var phi_22966_: bool;
    var phi_22998_: bool;
    var phi_23014_: bool;
    var phi_23030_: bool;
    var phi_23046_: bool;
    var phi_23078_: bool;
    var phi_23094_: bool;
    var phi_23110_: bool;
    var phi_23126_: bool;
    var phi_23158_: bool;
    var phi_23174_: bool;
    var phi_23190_: bool;
    var phi_23206_: bool;
    var phi_23230_: type_15_;
    var phi_23231_: type_15_;
    var phi_23254_: type_15_;
    var phi_23279_: bool;
    var phi_23296_: f32;
    var phi_23321_: bool;
    var phi_23338_: f32;
    var phi_35412_: bool;
    var phi_36251_: bool;
    var phi_36250_: bool;
    var phi_36249_: bool;
    var phi_36248_: bool;
    var phi_36247_: bool;
    var phi_36246_: bool;
    var phi_36253_: bool;
    var phi_36252_: bool;
    var local_49_: u32;
    var local_50_: u32;
    var local_51_: f32;
    var local_52_: bool;

    switch bitcast<i32>(0u) {
        default: {
            let _e152_ = wg_id_1_;
            let _e153_ = num_wg_1_;
            let _e154_ = lid_1_;
            let _e156_ = arrayLength((&core_inputs.member));
            let _e158_ = arrayLength((&quad_params.member));
            let _e160_ = arrayLength((&tile_checkpoints.member));
            let _e162_ = arrayLength((&feature_outputs.member));
            let _e163_ = (_e154_ & 255u);
            let _e164_ = (_e163_ & 127u);
            let _e167_ = ((_e163_ >> bitcast<u32>(7i)) != 0u);
            let _e172_ = select(1u, _e153_.y, (1u < _e153_.y));
            let _e176_ = uniforms.member.member_2_;
            let _e178_ = select(1u, _e176_, (1u < _e176_));
            let _e181_ = uniforms.member.member_7_;
            let _e182_ = (_e181_ == 4294967295u);
            if _e182_ {
                phi_303_ = select(16u, _e152_.x, (_e152_.x < _e178_));
            } else {
                let _e183_ = (_e152_.x < 4u);
                if _e183_ {
                    local_8_ = array<array<u32, 4>, 4>(array<u32, 4>(0u, 4u, 6u, 8u), array<u32, 4>(1u, 5u, 9u, 16u), array<u32, 4>(2u, 7u, 16u, 16u), array<u32, 4>(3u, 16u, 16u, 16u));
                    let _e184_ = (_e181_ & 3u);
                    if (_e184_ < 4u) {
                    } else {
                        break;
                    }
                    if _e183_ {
                    } else {
                        break;
                    }
                    let _e188_ = local_8_[_e184_][_e152_.x];
                    phi_302_ = select(16u, _e188_, (_e188_ < _e178_));
                } else {
                    phi_302_ = 16u;
                }
                let _e192_ = phi_302_;
                phi_303_ = _e192_;
            }
            let _e196_ = phi_303_;
            if (_e196_ >= _e178_) {
            } else {
                let _e199_ = ((_e152_.z * _e178_) + _e196_);
                if _e182_ {
                    phi_320_ = false;
                } else {
                    phi_320_ = ((_e181_ & 1073741824u) != 0u);
                }
                let _e203_ = phi_320_;
                if _e182_ {
                    phi_328_ = false;
                } else {
                    phi_328_ = ((_e181_ & 3u) == 0u);
                }
                let _e207_ = phi_328_;
                let _e210_ = uniforms.member.member;
                if (_e210_ > 0u) {
                    phi_338_ = (((_e210_ - 1u) >> bitcast<u32>(8i)) + 1u);
                } else {
                    phi_338_ = 0u;
                }
                let _e217_ = phi_338_;
                let _e220_ = uniforms.member.member_1_;
                let _e222_ = select(1u, _e220_, (1u < _e220_));
                let _e224_ = select(_e220_, _e217_, (_e217_ < _e220_));
                let _e227_ = uniforms.member.member_4_;
                let _e229_ = select(_e227_, _e224_, (_e224_ < _e227_));
                let _e230_ = (_e172_ <= 1u);
                if _e230_ {
                    phi_436_ = _e229_;
                    phi_437_ = _e224_;
                } else {
                    if (_e224_ > _e229_) {
                        phi_369_ = (_e224_ - _e229_);
                    } else {
                        phi_369_ = 0u;
                    }
                    let _e234_ = phi_369_;
                    if (_e234_ == 0u) {
                        phi_422_ = 0u;
                    } else {
                        if _e230_ {
                            phi_421_ = _e234_;
                        } else {
                            let _e236_ = (_e234_ - 1u);
                            phi_380_ = 1u;
                            phi_383_ = _e172_;
                            loop {
                                let _e238_ = phi_380_;
                                let _e240_ = phi_383_;
                                local_49_ = _e238_;
                                local_50_ = _e240_;
                                let _e243_ = (_e240_ <= (_e236_ >> bitcast<u32>(1i)));
                                if _e243_ {
                                    phi_381_ = (_e238_ << bitcast<u32>(1i));
                                    phi_384_ = (_e240_ << bitcast<u32>(1i));
                                } else {
                                    phi_381_ = u32();
                                    phi_384_ = u32();
                                }
                                let _e249_ = phi_381_;
                                let _e251_ = phi_384_;
                                phi_380_ = _e249_;
                                phi_383_ = _e251_;
                                if !(_e243_) {
                                    break;
                                }
                            }
                            phi_395_ = 0u;
                            let _e9380_ = local_49_;
                            phi_398_ = _e9380_;
                            phi_400_ = _e236_;
                            let _e9384_ = local_50_;
                            phi_402_ = _e9384_;
                            loop {
                                let _e254_ = phi_395_;
                                let _e256_ = phi_398_;
                                let _e258_ = phi_400_;
                                let _e260_ = phi_402_;
                                local_21_ = _e254_;
                                let _e261_ = (_e256_ > 0u);
                                if _e261_ {
                                    if (_e258_ >= _e260_) {
                                        phi_416_ = (_e254_ | _e256_);
                                        phi_417_ = (_e258_ - _e260_);
                                    } else {
                                        phi_416_ = _e254_;
                                        phi_417_ = _e258_;
                                    }
                                    let _e266_ = phi_416_;
                                    let _e268_ = phi_417_;
                                    phi_396_ = _e266_;
                                    phi_399_ = (_e256_ >> bitcast<u32>(1i));
                                    phi_401_ = _e268_;
                                    phi_403_ = (_e260_ >> bitcast<u32>(1i));
                                } else {
                                    phi_396_ = u32();
                                    phi_399_ = u32();
                                    phi_401_ = u32();
                                    phi_403_ = u32();
                                }
                                let _e274_ = phi_396_;
                                let _e276_ = phi_399_;
                                let _e278_ = phi_401_;
                                let _e280_ = phi_403_;
                                phi_395_ = _e274_;
                                phi_398_ = _e276_;
                                phi_400_ = _e278_;
                                phi_402_ = _e280_;
                                if !(_e261_) {
                                    break;
                                }
                            }
                            let _e283_ = local_21_;
                            phi_421_ = (_e283_ + 1u);
                        }
                        let _e286_ = phi_421_;
                        phi_422_ = _e286_;
                    }
                    let _e288_ = phi_422_;
                    let _e290_ = (_e229_ + (_e152_.y * _e288_));
                    let _e292_ = select(_e290_, _e224_, (_e224_ < _e290_));
                    let _e293_ = (_e292_ + _e288_);
                    phi_436_ = _e292_;
                    phi_437_ = select(_e293_, _e224_, (_e224_ < _e293_));
                }
                let _e297_ = phi_436_;
                let _e299_ = phi_437_;
                let _e300_ = (_e199_ * 25u);
                let _e301_ = (_e300_ < _e158_);
                if _e301_ {
                    if _e301_ {
                    } else {
                        break;
                    }
                    let _e304_ = quad_params.member[_e300_];
                    phi_451_ = _e304_;
                } else {
                    phi_451_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e306_ = phi_451_;
                let _e307_ = (_e300_ + 1u);
                let _e308_ = (_e307_ < _e158_);
                if _e308_ {
                    if _e308_ {
                    } else {
                        break;
                    }
                    let _e311_ = quad_params.member[_e307_];
                    phi_464_ = _e311_;
                } else {
                    phi_464_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e313_ = phi_464_;
                let _e314_ = (_e300_ + 2u);
                let _e315_ = (_e314_ < _e158_);
                if _e315_ {
                    if _e315_ {
                    } else {
                        break;
                    }
                    let _e318_ = quad_params.member[_e314_];
                    phi_477_ = _e318_;
                } else {
                    phi_477_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e320_ = phi_477_;
                let _e321_ = (_e300_ + 3u);
                let _e322_ = (_e321_ < _e158_);
                if _e322_ {
                    if _e322_ {
                    } else {
                        break;
                    }
                    let _e325_ = quad_params.member[_e321_];
                    phi_490_ = _e325_;
                } else {
                    phi_490_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e327_ = phi_490_;
                let _e328_ = (_e300_ + 4u);
                let _e329_ = (_e328_ < _e158_);
                if _e329_ {
                    if _e329_ {
                    } else {
                        break;
                    }
                    let _e332_ = quad_params.member[_e328_];
                    phi_503_ = _e332_;
                } else {
                    phi_503_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e334_ = phi_503_;
                let _e335_ = (_e300_ + 5u);
                let _e336_ = (_e335_ < _e158_);
                if _e336_ {
                    if _e336_ {
                    } else {
                        break;
                    }
                    let _e339_ = quad_params.member[_e335_];
                    phi_516_ = _e339_;
                } else {
                    phi_516_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e341_ = phi_516_;
                let _e342_ = (_e300_ + 6u);
                let _e343_ = (_e342_ < _e158_);
                if _e343_ {
                    if _e343_ {
                    } else {
                        break;
                    }
                    let _e346_ = quad_params.member[_e342_];
                    phi_529_ = _e346_;
                } else {
                    phi_529_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e348_ = phi_529_;
                let _e349_ = (_e300_ + 7u);
                let _e350_ = (_e349_ < _e158_);
                if _e350_ {
                    if _e350_ {
                    } else {
                        break;
                    }
                    let _e353_ = quad_params.member[_e349_];
                    phi_542_ = _e353_;
                } else {
                    phi_542_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e355_ = phi_542_;
                let _e356_ = (_e300_ + 8u);
                let _e357_ = (_e356_ < _e158_);
                if _e357_ {
                    if _e357_ {
                    } else {
                        break;
                    }
                    let _e360_ = quad_params.member[_e356_];
                    phi_555_ = _e360_;
                } else {
                    phi_555_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e362_ = phi_555_;
                let _e363_ = (_e300_ + 9u);
                let _e364_ = (_e363_ < _e158_);
                if _e364_ {
                    if _e364_ {
                    } else {
                        break;
                    }
                    let _e367_ = quad_params.member[_e363_];
                    phi_568_ = _e367_;
                } else {
                    phi_568_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e369_ = phi_568_;
                let _e370_ = (_e300_ + 10u);
                let _e371_ = (_e370_ < _e158_);
                if _e371_ {
                    if _e371_ {
                    } else {
                        break;
                    }
                    let _e374_ = quad_params.member[_e370_];
                    phi_581_ = _e374_;
                } else {
                    phi_581_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e376_ = phi_581_;
                let _e377_ = (_e300_ + 11u);
                let _e378_ = (_e377_ < _e158_);
                if _e378_ {
                    if _e378_ {
                    } else {
                        break;
                    }
                    let _e381_ = quad_params.member[_e377_];
                    phi_594_ = _e381_;
                } else {
                    phi_594_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e383_ = phi_594_;
                let _e384_ = (_e300_ + 12u);
                let _e385_ = (_e384_ < _e158_);
                if _e385_ {
                    if _e385_ {
                    } else {
                        break;
                    }
                    let _e388_ = quad_params.member[_e384_];
                    phi_607_ = _e388_;
                } else {
                    phi_607_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e390_ = phi_607_;
                let _e391_ = (_e300_ + 13u);
                let _e392_ = (_e391_ < _e158_);
                if _e392_ {
                    if _e392_ {
                    } else {
                        break;
                    }
                    let _e395_ = quad_params.member[_e391_];
                    phi_620_ = _e395_;
                } else {
                    phi_620_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e397_ = phi_620_;
                let _e398_ = (_e300_ + 14u);
                let _e399_ = (_e398_ < _e158_);
                if _e399_ {
                    if _e399_ {
                    } else {
                        break;
                    }
                    let _e402_ = quad_params.member[_e398_];
                    phi_633_ = _e402_;
                } else {
                    phi_633_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e404_ = phi_633_;
                let _e405_ = (_e300_ + 15u);
                let _e406_ = (_e405_ < _e158_);
                if _e406_ {
                    if _e406_ {
                    } else {
                        break;
                    }
                    let _e409_ = quad_params.member[_e405_];
                    phi_646_ = _e409_;
                } else {
                    phi_646_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e411_ = phi_646_;
                let _e412_ = (_e300_ + 16u);
                let _e413_ = (_e412_ < _e158_);
                if _e413_ {
                    if _e413_ {
                    } else {
                        break;
                    }
                    let _e416_ = quad_params.member[_e412_];
                    phi_659_ = _e416_;
                } else {
                    phi_659_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e418_ = phi_659_;
                let _e419_ = (_e300_ + 17u);
                let _e420_ = (_e419_ < _e158_);
                if _e420_ {
                    if _e420_ {
                    } else {
                        break;
                    }
                    let _e423_ = quad_params.member[_e419_];
                    phi_672_ = _e423_;
                } else {
                    phi_672_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e425_ = phi_672_;
                let _e426_ = (_e300_ + 18u);
                let _e427_ = (_e426_ < _e158_);
                if _e427_ {
                    if _e427_ {
                    } else {
                        break;
                    }
                    let _e430_ = quad_params.member[_e426_];
                    phi_685_ = _e430_;
                } else {
                    phi_685_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e432_ = phi_685_;
                let _e433_ = (_e300_ + 19u);
                let _e434_ = (_e433_ < _e158_);
                if _e434_ {
                    if _e434_ {
                    } else {
                        break;
                    }
                    let _e437_ = quad_params.member[_e433_];
                    phi_698_ = _e437_;
                } else {
                    phi_698_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e439_ = phi_698_;
                let _e440_ = (_e300_ + 20u);
                let _e441_ = (_e440_ < _e158_);
                if _e441_ {
                    if _e441_ {
                    } else {
                        break;
                    }
                    let _e444_ = quad_params.member[_e440_];
                    phi_711_ = _e444_;
                } else {
                    phi_711_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e446_ = phi_711_;
                let _e447_ = (_e300_ + 21u);
                let _e448_ = (_e447_ < _e158_);
                if _e448_ {
                    if _e448_ {
                    } else {
                        break;
                    }
                    let _e451_ = quad_params.member[_e447_];
                    phi_724_ = _e451_;
                } else {
                    phi_724_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e453_ = phi_724_;
                let _e454_ = (_e300_ + 22u);
                let _e455_ = (_e454_ < _e158_);
                if _e455_ {
                    if _e455_ {
                    } else {
                        break;
                    }
                    let _e458_ = quad_params.member[_e454_];
                    phi_737_ = _e458_;
                } else {
                    phi_737_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e460_ = phi_737_;
                let _e462_ = ((_e300_ + 23u) < _e158_);
                if _e462_ {
                    if _e462_ {
                    } else {
                        break;
                    }
                }
                let _e463_ = (_e300_ + 24u);
                let _e464_ = (_e463_ < _e158_);
                if _e464_ {
                    if _e464_ {
                    } else {
                        break;
                    }
                    let _e467_ = quad_params.member[_e463_];
                    phi_763_ = _e467_;
                } else {
                    phi_763_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e469_ = phi_763_;
                let _e471_ = bitcast<u32>(_e453_.x);
                let _e472_ = (_e471_ & 2139095040u);
                if (_e472_ != 2139095040u) {
                    if (_e453_.x <= 0f) {
                        phi_797_ = 0f;
                    } else {
                        if (_e453_.x >= 1f) {
                            phi_796_ = 1f;
                        } else {
                            if (_e472_ == 2139095040u) {
                                phi_790_ = true;
                            } else {
                                phi_790_ = (_e472_ == 0u);
                            }
                            let _e481_ = phi_790_;
                            phi_796_ = bitcast<f32>(select(_e471_, 0u, _e481_));
                        }
                        let _e485_ = phi_796_;
                        phi_797_ = _e485_;
                    }
                    let _e487_ = phi_797_;
                    phi_804_ = _e487_;
                } else {
                    phi_804_ = select(0f, 1f, (_e471_ == 2139095040u));
                }
                let _e489_ = phi_804_;
                let _e491_ = bitcast<u32>(_e453_.y);
                let _e492_ = (_e491_ & 2139095040u);
                if (_e492_ != 2139095040u) {
                    if (_e453_.y <= 0f) {
                        phi_837_ = 0f;
                    } else {
                        if (_e453_.y >= 1f) {
                            phi_836_ = 1f;
                        } else {
                            if (_e492_ == 2139095040u) {
                                phi_830_ = true;
                            } else {
                                phi_830_ = (_e492_ == 0u);
                            }
                            let _e501_ = phi_830_;
                            phi_836_ = bitcast<f32>(select(_e491_, 0u, _e501_));
                        }
                        let _e505_ = phi_836_;
                        phi_837_ = _e505_;
                    }
                    let _e507_ = phi_837_;
                    phi_844_ = _e507_;
                } else {
                    phi_844_ = select(0f, 1f, (_e491_ == 2139095040u));
                }
                let _e509_ = phi_844_;
                let _e511_ = bitcast<u32>(_e453_.z);
                let _e512_ = (_e511_ & 2139095040u);
                if (_e512_ != 2139095040u) {
                    if (_e453_.z <= 0f) {
                        phi_877_ = 0f;
                    } else {
                        if (_e453_.z >= 1f) {
                            phi_876_ = 1f;
                        } else {
                            if (_e512_ == 2139095040u) {
                                phi_870_ = true;
                            } else {
                                phi_870_ = (_e512_ == 0u);
                            }
                            let _e521_ = phi_870_;
                            phi_876_ = bitcast<f32>(select(_e511_, 0u, _e521_));
                        }
                        let _e525_ = phi_876_;
                        phi_877_ = _e525_;
                    }
                    let _e527_ = phi_877_;
                    phi_884_ = _e527_;
                } else {
                    phi_884_ = select(0f, 1f, (_e511_ == 2139095040u));
                }
                let _e529_ = phi_884_;
                let _e531_ = bitcast<u32>(_e453_.w);
                let _e532_ = (_e531_ & 2139095040u);
                if (_e532_ != 2139095040u) {
                    if (_e453_.w <= 0f) {
                        phi_917_ = 0f;
                    } else {
                        if (_e453_.w >= 1f) {
                            phi_916_ = 1f;
                        } else {
                            if (_e532_ == 2139095040u) {
                                phi_910_ = true;
                            } else {
                                phi_910_ = (_e532_ == 0u);
                            }
                            let _e541_ = phi_910_;
                            phi_916_ = bitcast<f32>(select(_e531_, 0u, _e541_));
                        }
                        let _e545_ = phi_916_;
                        phi_917_ = _e545_;
                    }
                    let _e547_ = phi_917_;
                    phi_924_ = _e547_;
                } else {
                    phi_924_ = select(0f, 1f, (_e531_ == 2139095040u));
                }
                let _e549_ = phi_924_;
                let _e551_ = bitcast<u32>(_e411_.x);
                let _e552_ = (_e551_ & 2139095040u);
                if (_e552_ != 2139095040u) {
                    if (_e411_.x <= 0f) {
                        phi_957_ = 0f;
                    } else {
                        if (_e411_.x >= 1f) {
                            phi_956_ = 1f;
                        } else {
                            if (_e552_ == 2139095040u) {
                                phi_950_ = true;
                            } else {
                                phi_950_ = (_e552_ == 0u);
                            }
                            let _e561_ = phi_950_;
                            phi_956_ = bitcast<f32>(select(_e551_, 0u, _e561_));
                        }
                        let _e565_ = phi_956_;
                        phi_957_ = _e565_;
                    }
                    let _e567_ = phi_957_;
                    phi_964_ = _e567_;
                } else {
                    phi_964_ = select(0f, 1f, (_e551_ == 2139095040u));
                }
                let _e569_ = phi_964_;
                let _e571_ = bitcast<u32>(_e411_.y);
                let _e572_ = (_e571_ & 2139095040u);
                if (_e572_ != 2139095040u) {
                    if (_e411_.y <= 0f) {
                        phi_997_ = 0f;
                    } else {
                        if (_e411_.y >= 1f) {
                            phi_996_ = 1f;
                        } else {
                            if (_e572_ == 2139095040u) {
                                phi_990_ = true;
                            } else {
                                phi_990_ = (_e572_ == 0u);
                            }
                            let _e581_ = phi_990_;
                            phi_996_ = bitcast<f32>(select(_e571_, 0u, _e581_));
                        }
                        let _e585_ = phi_996_;
                        phi_997_ = _e585_;
                    }
                    let _e587_ = phi_997_;
                    phi_1004_ = _e587_;
                } else {
                    phi_1004_ = select(0f, 1f, (_e571_ == 2139095040u));
                }
                let _e589_ = phi_1004_;
                let _e591_ = bitcast<u32>(_e411_.z);
                let _e592_ = (_e591_ & 2139095040u);
                if (_e592_ != 2139095040u) {
                    if (_e411_.z <= 0f) {
                        phi_1037_ = 0f;
                    } else {
                        if (_e411_.z >= 1f) {
                            phi_1036_ = 1f;
                        } else {
                            if (_e592_ == 2139095040u) {
                                phi_1030_ = true;
                            } else {
                                phi_1030_ = (_e592_ == 0u);
                            }
                            let _e601_ = phi_1030_;
                            phi_1036_ = bitcast<f32>(select(_e591_, 0u, _e601_));
                        }
                        let _e605_ = phi_1036_;
                        phi_1037_ = _e605_;
                    }
                    let _e607_ = phi_1037_;
                    phi_1044_ = _e607_;
                } else {
                    phi_1044_ = select(0f, 1f, (_e591_ == 2139095040u));
                }
                let _e609_ = phi_1044_;
                let _e611_ = bitcast<u32>(_e411_.w);
                let _e612_ = (_e611_ & 2139095040u);
                if (_e612_ != 2139095040u) {
                    if (_e411_.w <= 0f) {
                        phi_1077_ = 0f;
                    } else {
                        if (_e411_.w >= 1f) {
                            phi_1076_ = 1f;
                        } else {
                            if (_e612_ == 2139095040u) {
                                phi_1070_ = true;
                            } else {
                                phi_1070_ = (_e612_ == 0u);
                            }
                            let _e621_ = phi_1070_;
                            phi_1076_ = bitcast<f32>(select(_e611_, 0u, _e621_));
                        }
                        let _e625_ = phi_1076_;
                        phi_1077_ = _e625_;
                    }
                    let _e627_ = phi_1077_;
                    phi_1084_ = _e627_;
                } else {
                    phi_1084_ = select(0f, 1f, (_e611_ == 2139095040u));
                }
                let _e629_ = phi_1084_;
                if _e230_ {
                    let _e634_ = (_e229_ > 0u);
                    if _e634_ {
                        let _e635_ = (_e229_ < _e222_);
                        let _e636_ = (_e163_ == 0u);
                        if _e636_ {
                            if _e635_ {
                                let _e639_ = (((_e199_ * _e222_) + _e229_) * 8u);
                                let _e640_ = (_e639_ < _e160_);
                                if _e640_ {
                                    if _e640_ {
                                    } else {
                                        break;
                                    }
                                    let _e643_ = tile_checkpoints.member[_e639_];
                                    phi_1120_ = _e643_;
                                } else {
                                    phi_1120_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e645_ = phi_1120_;
                                let _e647_ = bitcast<u32>(_e645_.x);
                                let _e648_ = (_e647_ & 2139095040u);
                                if (_e648_ == 2139095040u) {
                                    phi_1132_ = true;
                                } else {
                                    phi_1132_ = (_e648_ == 0u);
                                }
                                let _e652_ = phi_1132_;
                                let _e656_ = bitcast<u32>(_e645_.y);
                                let _e657_ = (_e656_ & 2139095040u);
                                if (_e657_ == 2139095040u) {
                                    phi_1149_ = true;
                                } else {
                                    phi_1149_ = (_e657_ == 0u);
                                }
                                let _e661_ = phi_1149_;
                                let _e665_ = bitcast<u32>(_e645_.z);
                                let _e666_ = (_e665_ & 2139095040u);
                                if (_e666_ == 2139095040u) {
                                    phi_1166_ = true;
                                } else {
                                    phi_1166_ = (_e666_ == 0u);
                                }
                                let _e670_ = phi_1166_;
                                let _e674_ = bitcast<u32>(_e645_.w);
                                let _e675_ = (_e674_ & 2139095040u);
                                if (_e675_ == 2139095040u) {
                                    phi_1183_ = true;
                                } else {
                                    phi_1183_ = (_e675_ == 0u);
                                }
                                let _e679_ = phi_1183_;
                                local = array<f32, 4>(bitcast<f32>(select(_e647_, 0u, _e652_)), bitcast<f32>(select(_e656_, 0u, _e661_)), bitcast<f32>(select(_e665_, 0u, _e670_)), bitcast<f32>(select(_e674_, 0u, _e679_)));
                                let _e683_ = (_e639_ + 1u);
                                let _e684_ = (_e683_ < _e160_);
                                if _e684_ {
                                    if _e684_ {
                                    } else {
                                        break;
                                    }
                                    let _e687_ = tile_checkpoints.member[_e683_];
                                    phi_1207_ = _e687_;
                                } else {
                                    phi_1207_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e689_ = phi_1207_;
                                let _e691_ = bitcast<u32>(_e689_.x);
                                let _e692_ = (_e691_ & 2139095040u);
                                if (_e692_ == 2139095040u) {
                                    phi_1219_ = true;
                                } else {
                                    phi_1219_ = (_e692_ == 0u);
                                }
                                let _e696_ = phi_1219_;
                                let _e700_ = bitcast<u32>(_e689_.y);
                                let _e701_ = (_e700_ & 2139095040u);
                                if (_e701_ == 2139095040u) {
                                    phi_1236_ = true;
                                } else {
                                    phi_1236_ = (_e701_ == 0u);
                                }
                                let _e705_ = phi_1236_;
                                let _e709_ = bitcast<u32>(_e689_.z);
                                let _e710_ = (_e709_ & 2139095040u);
                                if (_e710_ == 2139095040u) {
                                    phi_1253_ = true;
                                } else {
                                    phi_1253_ = (_e710_ == 0u);
                                }
                                let _e714_ = phi_1253_;
                                let _e718_ = bitcast<u32>(_e689_.w);
                                let _e719_ = (_e718_ & 2139095040u);
                                if (_e719_ == 2139095040u) {
                                    phi_1270_ = true;
                                } else {
                                    phi_1270_ = (_e719_ == 0u);
                                }
                                let _e723_ = phi_1270_;
                                local_1_ = array<f32, 4>(bitcast<f32>(select(_e691_, 0u, _e696_)), bitcast<f32>(select(_e700_, 0u, _e705_)), bitcast<f32>(select(_e709_, 0u, _e714_)), bitcast<f32>(select(_e718_, 0u, _e723_)));
                                let _e727_ = (_e639_ + 2u);
                                let _e728_ = (_e727_ < _e160_);
                                if _e728_ {
                                    if _e728_ {
                                    } else {
                                        break;
                                    }
                                    let _e731_ = tile_checkpoints.member[_e727_];
                                    phi_1294_ = _e731_;
                                } else {
                                    phi_1294_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e733_ = phi_1294_;
                                let _e735_ = bitcast<u32>(_e733_.x);
                                let _e736_ = (_e735_ & 2139095040u);
                                if (_e736_ == 2139095040u) {
                                    phi_1306_ = true;
                                } else {
                                    phi_1306_ = (_e736_ == 0u);
                                }
                                let _e740_ = phi_1306_;
                                let _e744_ = bitcast<u32>(_e733_.y);
                                let _e745_ = (_e744_ & 2139095040u);
                                if (_e745_ == 2139095040u) {
                                    phi_1323_ = true;
                                } else {
                                    phi_1323_ = (_e745_ == 0u);
                                }
                                let _e749_ = phi_1323_;
                                let _e753_ = bitcast<u32>(_e733_.z);
                                let _e754_ = (_e753_ & 2139095040u);
                                if (_e754_ == 2139095040u) {
                                    phi_1340_ = true;
                                } else {
                                    phi_1340_ = (_e754_ == 0u);
                                }
                                let _e758_ = phi_1340_;
                                let _e762_ = bitcast<u32>(_e733_.w);
                                let _e763_ = (_e762_ & 2139095040u);
                                if (_e763_ == 2139095040u) {
                                    phi_1357_ = true;
                                } else {
                                    phi_1357_ = (_e763_ == 0u);
                                }
                                let _e767_ = phi_1357_;
                                local_2_ = array<f32, 4>(bitcast<f32>(select(_e735_, 0u, _e740_)), bitcast<f32>(select(_e744_, 0u, _e749_)), bitcast<f32>(select(_e753_, 0u, _e758_)), bitcast<f32>(select(_e762_, 0u, _e767_)));
                                let _e771_ = (_e639_ + 3u);
                                let _e772_ = (_e771_ < _e160_);
                                if _e772_ {
                                    if _e772_ {
                                    } else {
                                        break;
                                    }
                                    let _e775_ = tile_checkpoints.member[_e771_];
                                    phi_1381_ = _e775_;
                                } else {
                                    phi_1381_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e777_ = phi_1381_;
                                let _e779_ = bitcast<u32>(_e777_.x);
                                let _e780_ = (_e779_ & 2139095040u);
                                if (_e780_ == 2139095040u) {
                                    phi_1393_ = true;
                                } else {
                                    phi_1393_ = (_e780_ == 0u);
                                }
                                let _e784_ = phi_1393_;
                                let _e788_ = bitcast<u32>(_e777_.y);
                                let _e789_ = (_e788_ & 2139095040u);
                                if (_e789_ == 2139095040u) {
                                    phi_1410_ = true;
                                } else {
                                    phi_1410_ = (_e789_ == 0u);
                                }
                                let _e793_ = phi_1410_;
                                let _e797_ = bitcast<u32>(_e777_.z);
                                let _e798_ = (_e797_ & 2139095040u);
                                if (_e798_ == 2139095040u) {
                                    phi_1427_ = true;
                                } else {
                                    phi_1427_ = (_e798_ == 0u);
                                }
                                let _e802_ = phi_1427_;
                                let _e806_ = bitcast<u32>(_e777_.w);
                                let _e807_ = (_e806_ & 2139095040u);
                                if (_e807_ == 2139095040u) {
                                    phi_1444_ = true;
                                } else {
                                    phi_1444_ = (_e807_ == 0u);
                                }
                                let _e811_ = phi_1444_;
                                local_3_ = array<f32, 4>(bitcast<f32>(select(_e779_, 0u, _e784_)), bitcast<f32>(select(_e788_, 0u, _e793_)), bitcast<f32>(select(_e797_, 0u, _e802_)), bitcast<f32>(select(_e806_, 0u, _e811_)));
                                let _e815_ = (_e639_ + 4u);
                                let _e816_ = (_e815_ < _e160_);
                                if _e816_ {
                                    if _e816_ {
                                    } else {
                                        break;
                                    }
                                    let _e819_ = tile_checkpoints.member[_e815_];
                                    phi_1468_ = _e819_;
                                } else {
                                    phi_1468_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e821_ = phi_1468_;
                                let _e823_ = bitcast<u32>(_e821_.x);
                                let _e824_ = (_e823_ & 2139095040u);
                                if (_e824_ == 2139095040u) {
                                    phi_1480_ = true;
                                } else {
                                    phi_1480_ = (_e824_ == 0u);
                                }
                                let _e828_ = phi_1480_;
                                let _e832_ = bitcast<u32>(_e821_.y);
                                let _e833_ = (_e832_ & 2139095040u);
                                if (_e833_ == 2139095040u) {
                                    phi_1497_ = true;
                                } else {
                                    phi_1497_ = (_e833_ == 0u);
                                }
                                let _e837_ = phi_1497_;
                                let _e841_ = bitcast<u32>(_e821_.z);
                                let _e842_ = (_e841_ & 2139095040u);
                                if (_e842_ == 2139095040u) {
                                    phi_1514_ = true;
                                } else {
                                    phi_1514_ = (_e842_ == 0u);
                                }
                                let _e846_ = phi_1514_;
                                let _e850_ = bitcast<u32>(_e821_.w);
                                let _e851_ = (_e850_ & 2139095040u);
                                if (_e851_ == 2139095040u) {
                                    phi_1531_ = true;
                                } else {
                                    phi_1531_ = (_e851_ == 0u);
                                }
                                let _e855_ = phi_1531_;
                                local_4_ = array<f32, 4>(bitcast<f32>(select(_e823_, 0u, _e828_)), bitcast<f32>(select(_e832_, 0u, _e837_)), bitcast<f32>(select(_e841_, 0u, _e846_)), bitcast<f32>(select(_e850_, 0u, _e855_)));
                                let _e859_ = (_e639_ + 5u);
                                let _e860_ = (_e859_ < _e160_);
                                if _e860_ {
                                    if _e860_ {
                                    } else {
                                        break;
                                    }
                                    let _e863_ = tile_checkpoints.member[_e859_];
                                    phi_1555_ = _e863_;
                                } else {
                                    phi_1555_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e865_ = phi_1555_;
                                let _e867_ = bitcast<u32>(_e865_.x);
                                let _e868_ = (_e867_ & 2139095040u);
                                if (_e868_ == 2139095040u) {
                                    phi_1567_ = true;
                                } else {
                                    phi_1567_ = (_e868_ == 0u);
                                }
                                let _e872_ = phi_1567_;
                                let _e876_ = bitcast<u32>(_e865_.y);
                                let _e877_ = (_e876_ & 2139095040u);
                                if (_e877_ == 2139095040u) {
                                    phi_1584_ = true;
                                } else {
                                    phi_1584_ = (_e877_ == 0u);
                                }
                                let _e881_ = phi_1584_;
                                let _e885_ = bitcast<u32>(_e865_.z);
                                let _e886_ = (_e885_ & 2139095040u);
                                if (_e886_ == 2139095040u) {
                                    phi_1601_ = true;
                                } else {
                                    phi_1601_ = (_e886_ == 0u);
                                }
                                let _e890_ = phi_1601_;
                                let _e894_ = bitcast<u32>(_e865_.w);
                                let _e895_ = (_e894_ & 2139095040u);
                                if (_e895_ == 2139095040u) {
                                    phi_1618_ = true;
                                } else {
                                    phi_1618_ = (_e895_ == 0u);
                                }
                                let _e899_ = phi_1618_;
                                local_5_ = array<f32, 4>(bitcast<f32>(select(_e867_, 0u, _e872_)), bitcast<f32>(select(_e876_, 0u, _e881_)), bitcast<f32>(select(_e885_, 0u, _e890_)), bitcast<f32>(select(_e894_, 0u, _e899_)));
                                let _e903_ = (_e639_ + 6u);
                                let _e904_ = (_e903_ < _e160_);
                                if _e904_ {
                                    if _e904_ {
                                    } else {
                                        break;
                                    }
                                    let _e907_ = tile_checkpoints.member[_e903_];
                                    phi_1642_ = _e907_;
                                } else {
                                    phi_1642_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e909_ = phi_1642_;
                                let _e911_ = bitcast<u32>(_e909_.x);
                                let _e912_ = (_e911_ & 2139095040u);
                                if (_e912_ == 2139095040u) {
                                    phi_1654_ = true;
                                } else {
                                    phi_1654_ = (_e912_ == 0u);
                                }
                                let _e916_ = phi_1654_;
                                let _e918_ = bitcast<f32>(select(_e911_, 0u, _e916_));
                                if (_e918_ < -1000000000000000000f) {
                                    phi_1671_ = -1000000000000000000f;
                                } else {
                                    phi_1671_ = select(_e918_, 1000000000000000000f, (_e918_ > 1000000000000000000f));
                                }
                                let _e923_ = phi_1671_;
                                let _e925_ = bitcast<u32>(_e909_.y);
                                let _e926_ = (_e925_ & 2139095040u);
                                if (_e926_ == 2139095040u) {
                                    phi_1683_ = true;
                                } else {
                                    phi_1683_ = (_e926_ == 0u);
                                }
                                let _e930_ = phi_1683_;
                                let _e932_ = bitcast<f32>(select(_e925_, 0u, _e930_));
                                if (_e932_ < -1000000000000000000f) {
                                    phi_1700_ = -1000000000000000000f;
                                } else {
                                    phi_1700_ = select(_e932_, 1000000000000000000f, (_e932_ > 1000000000000000000f));
                                }
                                let _e937_ = phi_1700_;
                                let _e939_ = bitcast<u32>(_e909_.z);
                                let _e940_ = (_e939_ & 2139095040u);
                                if (_e940_ == 2139095040u) {
                                    phi_1712_ = true;
                                } else {
                                    phi_1712_ = (_e940_ == 0u);
                                }
                                let _e944_ = phi_1712_;
                                let _e946_ = bitcast<f32>(select(_e939_, 0u, _e944_));
                                if (_e946_ < -1000000000000000000f) {
                                    phi_1729_ = -1000000000000000000f;
                                } else {
                                    phi_1729_ = select(_e946_, 1000000000000000000f, (_e946_ > 1000000000000000000f));
                                }
                                let _e951_ = phi_1729_;
                                let _e953_ = bitcast<u32>(_e909_.w);
                                let _e954_ = (_e953_ & 2139095040u);
                                if (_e954_ == 2139095040u) {
                                    phi_1741_ = true;
                                } else {
                                    phi_1741_ = (_e954_ == 0u);
                                }
                                let _e958_ = phi_1741_;
                                let _e960_ = bitcast<f32>(select(_e953_, 0u, _e958_));
                                if (_e960_ < -1000000000000000000f) {
                                    phi_1758_ = -1000000000000000000f;
                                } else {
                                    phi_1758_ = select(_e960_, 1000000000000000000f, (_e960_ > 1000000000000000000f));
                                }
                                let _e965_ = phi_1758_;
                                local_6_ = array<f32, 4>(_e923_, _e937_, _e951_, _e965_);
                                let _e967_ = (_e639_ + 7u);
                                let _e968_ = (_e967_ < _e160_);
                                if _e968_ {
                                    if _e968_ {
                                    } else {
                                        break;
                                    }
                                    let _e971_ = tile_checkpoints.member[_e967_];
                                    phi_1777_ = _e971_;
                                } else {
                                    phi_1777_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e973_ = phi_1777_;
                                let _e975_ = bitcast<u32>(_e973_.x);
                                let _e976_ = (_e975_ & 2139095040u);
                                if (_e976_ == 2139095040u) {
                                    phi_1789_ = true;
                                } else {
                                    phi_1789_ = (_e976_ == 0u);
                                }
                                let _e980_ = phi_1789_;
                                let _e982_ = bitcast<f32>(select(_e975_, 0u, _e980_));
                                if (_e982_ < -1000000000000000000f) {
                                    phi_1806_ = -1000000000000000000f;
                                } else {
                                    phi_1806_ = select(_e982_, 1000000000000000000f, (_e982_ > 1000000000000000000f));
                                }
                                let _e987_ = phi_1806_;
                                let _e989_ = bitcast<u32>(_e973_.y);
                                let _e990_ = (_e989_ & 2139095040u);
                                if (_e990_ == 2139095040u) {
                                    phi_1818_ = true;
                                } else {
                                    phi_1818_ = (_e990_ == 0u);
                                }
                                let _e994_ = phi_1818_;
                                let _e996_ = bitcast<f32>(select(_e989_, 0u, _e994_));
                                if (_e996_ < -1000000000000000000f) {
                                    phi_1835_ = -1000000000000000000f;
                                } else {
                                    phi_1835_ = select(_e996_, 1000000000000000000f, (_e996_ > 1000000000000000000f));
                                }
                                let _e1001_ = phi_1835_;
                                let _e1003_ = bitcast<u32>(_e973_.z);
                                let _e1004_ = (_e1003_ & 2139095040u);
                                if (_e1004_ == 2139095040u) {
                                    phi_1847_ = true;
                                } else {
                                    phi_1847_ = (_e1004_ == 0u);
                                }
                                let _e1008_ = phi_1847_;
                                let _e1010_ = bitcast<f32>(select(_e1003_, 0u, _e1008_));
                                if (_e1010_ < -1000000000000000000f) {
                                    phi_1864_ = -1000000000000000000f;
                                } else {
                                    phi_1864_ = select(_e1010_, 1000000000000000000f, (_e1010_ > 1000000000000000000f));
                                }
                                let _e1015_ = phi_1864_;
                                let _e1017_ = bitcast<u32>(_e973_.w);
                                let _e1018_ = (_e1017_ & 2139095040u);
                                if (_e1018_ == 2139095040u) {
                                    phi_1876_ = true;
                                } else {
                                    phi_1876_ = (_e1018_ == 0u);
                                }
                                let _e1022_ = phi_1876_;
                                let _e1024_ = bitcast<f32>(select(_e1017_, 0u, _e1022_));
                                if (_e1024_ < -1000000000000000000f) {
                                    phi_1893_ = -1000000000000000000f;
                                } else {
                                    phi_1893_ = select(_e1024_, 1000000000000000000f, (_e1024_ > 1000000000000000000f));
                                }
                                let _e1029_ = phi_1893_;
                                local_7_ = array<f32, 4>(_e987_, _e1001_, _e1015_, _e1029_);
                                phi_1902_ = type_15_(0u, 4u);
                                loop {
                                    let _e1032_ = phi_1902_;
                                    if (_e1032_.member < _e1032_.member_1_) {
                                        phi_1903_ = type_15_((_e1032_.member + 1u), _e1032_.member_1_);
                                        phi_1926_ = type_15_(1u, _e1032_.member);
                                    } else {
                                        phi_1903_ = _e1032_;
                                        phi_1926_ = type_15_(0u, u32());
                                    }
                                    let _e1045_ = phi_1903_;
                                    let _e1047_ = phi_1926_;
                                    let _e1051_ = (bitcast<i32>(_e1047_.member) != 0i);
                                    if _e1051_ {
                                        let _e1052_ = (_e1047_.member_1_ < 4u);
                                        if _e1052_ {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        let _e1054_ = local[_e1047_.member_1_];
                                        if (_e1047_.member_1_ < 256u) {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1047_.member_1_] = _e1054_;
                                        if _e1052_ {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        let _e1058_ = local_2_[_e1047_.member_1_];
                                        let _e1059_ = (4u + _e1047_.member_1_);
                                        if (_e1059_ < 256u) {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1059_] = _e1058_;
                                        if _e1052_ {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        let _e1063_ = local_1_[_e1047_.member_1_];
                                        let _e1064_ = (8u + _e1047_.member_1_);
                                        if (_e1064_ < 256u) {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1064_] = _e1063_;
                                        if _e1052_ {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        let _e1068_ = local_3_[_e1047_.member_1_];
                                        let _e1069_ = (12u + _e1047_.member_1_);
                                        if (_e1069_ < 256u) {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1069_] = _e1068_;
                                        if _e1052_ {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        let _e1073_ = local_4_[_e1047_.member_1_];
                                        let _e1074_ = (16u + _e1047_.member_1_);
                                        if (_e1074_ < 256u) {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1074_] = _e1073_;
                                        if _e1052_ {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        let _e1078_ = local_5_[_e1047_.member_1_];
                                        let _e1079_ = (20u + _e1047_.member_1_);
                                        if (_e1079_ < 256u) {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1079_] = _e1078_;
                                        if _e1052_ {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        let _e1083_ = local_6_[_e1047_.member_1_];
                                        let _e1084_ = (24u + _e1047_.member_1_);
                                        if (_e1084_ < 256u) {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1084_] = _e1083_;
                                        if _e1052_ {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        let _e1088_ = local_7_[_e1047_.member_1_];
                                        let _e1089_ = (28u + _e1047_.member_1_);
                                        if (_e1089_ < 256u) {
                                        } else {
                                            phi_33640_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1089_] = _e1088_;
                                    }
                                    phi_1902_ = _e1045_;
                                    phi_33640_ = false;
                                    if !(_e1051_) {
                                        break;
                                    }
                                }
                                let _e1094_ = phi_33640_;
                                if _e1094_ {
                                    break;
                                }
                                phi_33836_ = _e1094_;
                            } else {
                                phi_33836_ = false;
                            }
                            let _e1096_ = phi_33836_;
                            phi_33835_ = _e1096_;
                            phi_2046_ = select(true, false, _e635_);
                        } else {
                            phi_33835_ = false;
                            phi_2046_ = false;
                        }
                        let _e1099_ = phi_33835_;
                        let _e1101_ = phi_2046_;
                        phi_33834_ = _e1099_;
                        phi_2048_ = _e635_;
                        phi_2052_ = _e1101_;
                        phi_2053_ = select(false, _e635_, _e636_);
                    } else {
                        phi_33834_ = false;
                        phi_2048_ = bool();
                        phi_2052_ = false;
                        phi_2053_ = false;
                    }
                    let _e1104_ = phi_33834_;
                    let _e1106_ = phi_2048_;
                    let _e1108_ = phi_2052_;
                    let _e1110_ = phi_2053_;
                    phi_33833_ = _e1104_;
                    phi_2054_ = _e1106_;
                    phi_2057_ = select(true, false, _e634_);
                    phi_2058_ = _e1108_;
                    phi_2059_ = _e1110_;
                } else {
                    phi_33833_ = false;
                    phi_2054_ = bool();
                    phi_2057_ = true;
                    phi_2058_ = false;
                    phi_2059_ = false;
                }
                let _e1113_ = phi_33833_;
                let _e1115_ = phi_2054_;
                let _e1117_ = phi_2057_;
                let _e1119_ = phi_2058_;
                let _e1121_ = phi_2059_;
                if _e1117_ {
                    phi_2069_ = (_e163_ == 0u);
                } else {
                    phi_2069_ = _e1119_;
                }
                let _e1124_ = phi_2069_;
                let _e1125_ = select(_e1115_, false, _e1117_);
                if _e1124_ {
                    phi_2076_ = type_15_(0u, 32u);
                    loop {
                        let _e1128_ = phi_2076_;
                        if (_e1128_.member < _e1128_.member_1_) {
                            phi_2077_ = type_15_((_e1128_.member + 1u), _e1128_.member_1_);
                            phi_2100_ = type_15_(1u, _e1128_.member);
                        } else {
                            phi_2077_ = _e1128_;
                            phi_2100_ = type_15_(0u, u32());
                        }
                        let _e1141_ = phi_2077_;
                        let _e1143_ = phi_2100_;
                        let _e1147_ = (bitcast<i32>(_e1143_.member) != 0i);
                        if _e1147_ {
                            if (_e1143_.member_1_ < 256u) {
                            } else {
                                phi_33828_ = true;
                                break;
                            }
                            aux_scratch[_e1143_.member_1_] = 0f;
                        }
                        phi_2076_ = _e1141_;
                        phi_33828_ = _e1113_;
                        if !(_e1147_) {
                            break;
                        }
                    }
                    let _e1152_ = phi_33828_;
                    if _e1152_ {
                        break;
                    }
                    phi_33869_ = _e1152_;
                } else {
                    phi_33869_ = _e1113_;
                }
                let _e1154_ = phi_33869_;
                workgroupBarrier();
                phi_33866_ = _e1154_;
                phi_2126_ = type_15_(_e297_, _e299_);
                loop {
                    let _e1161_ = phi_33866_;
                    let _e1163_ = phi_2126_;
                    if (_e1163_.member < _e1163_.member_1_) {
                        phi_2127_ = type_15_((_e1163_.member + 1u), _e1163_.member_1_);
                        phi_2150_ = type_15_(1u, _e1163_.member);
                    } else {
                        phi_2127_ = _e1163_;
                        phi_2150_ = type_15_(0u, u32());
                    }
                    let _e1176_ = phi_2127_;
                    let _e1178_ = phi_2150_;
                    let _e1182_ = (bitcast<i32>(_e1178_.member) != 0i);
                    if _e1182_ {
                        let _e1184_ = (_e163_ == 0u);
                        if _e1184_ {
                            if select(select(_e1115_, _e1125_, _e1124_), _e1125_, select(true, false, select(select(_e1121_, false, _e1117_), true, _e1124_))) {
                                phi_2171_ = select(true, false, (_e1178_.member_1_ == _e229_));
                            } else {
                                phi_2171_ = true;
                            }
                            let _e1188_ = phi_2171_;
                            if _e1188_ {
                                let _e1191_ = (((_e199_ * _e222_) + _e1178_.member_1_) * 8u);
                                let _e1193_ = aux_scratch[0];
                                let _e1195_ = aux_scratch[1];
                                let _e1197_ = aux_scratch[2];
                                let _e1199_ = aux_scratch[3];
                                let _e1201_ = aux_scratch[4];
                                let _e1203_ = aux_scratch[5];
                                let _e1205_ = aux_scratch[6];
                                let _e1207_ = aux_scratch[7];
                                let _e1209_ = aux_scratch[8];
                                let _e1211_ = aux_scratch[9];
                                let _e1213_ = aux_scratch[10];
                                let _e1215_ = aux_scratch[11];
                                let _e1217_ = aux_scratch[12];
                                let _e1219_ = aux_scratch[13];
                                let _e1221_ = aux_scratch[14];
                                let _e1223_ = aux_scratch[15];
                                let _e1225_ = aux_scratch[16];
                                let _e1227_ = aux_scratch[17];
                                let _e1229_ = aux_scratch[18];
                                let _e1231_ = aux_scratch[19];
                                let _e1233_ = aux_scratch[20];
                                let _e1235_ = aux_scratch[21];
                                let _e1237_ = aux_scratch[22];
                                let _e1239_ = aux_scratch[23];
                                let _e1241_ = aux_scratch[24];
                                let _e1243_ = aux_scratch[25];
                                let _e1245_ = aux_scratch[26];
                                let _e1247_ = aux_scratch[27];
                                let _e1249_ = aux_scratch[28];
                                let _e1251_ = aux_scratch[29];
                                let _e1253_ = aux_scratch[30];
                                let _e1255_ = aux_scratch[31];
                                let _e1256_ = bitcast<u32>(_e1193_);
                                let _e1257_ = (_e1256_ & 2139095040u);
                                if (_e1257_ == 2139095040u) {
                                    phi_2252_ = true;
                                } else {
                                    phi_2252_ = (_e1257_ == 0u);
                                }
                                let _e1261_ = phi_2252_;
                                let _e1264_ = bitcast<u32>(_e1195_);
                                let _e1265_ = (_e1264_ & 2139095040u);
                                if (_e1265_ == 2139095040u) {
                                    phi_2268_ = true;
                                } else {
                                    phi_2268_ = (_e1265_ == 0u);
                                }
                                let _e1269_ = phi_2268_;
                                let _e1272_ = bitcast<u32>(_e1197_);
                                let _e1273_ = (_e1272_ & 2139095040u);
                                if (_e1273_ == 2139095040u) {
                                    phi_2284_ = true;
                                } else {
                                    phi_2284_ = (_e1273_ == 0u);
                                }
                                let _e1277_ = phi_2284_;
                                let _e1280_ = bitcast<u32>(_e1199_);
                                let _e1281_ = (_e1280_ & 2139095040u);
                                if (_e1281_ == 2139095040u) {
                                    phi_2300_ = true;
                                } else {
                                    phi_2300_ = (_e1281_ == 0u);
                                }
                                let _e1285_ = phi_2300_;
                                let _e1289_ = (_e1191_ < _e160_);
                                if _e1289_ {
                                    if _e1289_ {
                                    } else {
                                        phi_36252_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1191_] = vec4<f32>(bitcast<f32>(select(_e1256_, 0u, _e1261_)), bitcast<f32>(select(_e1264_, 0u, _e1269_)), bitcast<f32>(select(_e1272_, 0u, _e1277_)), bitcast<f32>(select(_e1280_, 0u, _e1285_)));
                                }
                                let _e1292_ = (_e1191_ + 1u);
                                let _e1293_ = bitcast<u32>(_e1209_);
                                let _e1294_ = (_e1293_ & 2139095040u);
                                if (_e1294_ == 2139095040u) {
                                    phi_2328_ = true;
                                } else {
                                    phi_2328_ = (_e1294_ == 0u);
                                }
                                let _e1298_ = phi_2328_;
                                let _e1301_ = bitcast<u32>(_e1211_);
                                let _e1302_ = (_e1301_ & 2139095040u);
                                if (_e1302_ == 2139095040u) {
                                    phi_2344_ = true;
                                } else {
                                    phi_2344_ = (_e1302_ == 0u);
                                }
                                let _e1306_ = phi_2344_;
                                let _e1309_ = bitcast<u32>(_e1213_);
                                let _e1310_ = (_e1309_ & 2139095040u);
                                if (_e1310_ == 2139095040u) {
                                    phi_2360_ = true;
                                } else {
                                    phi_2360_ = (_e1310_ == 0u);
                                }
                                let _e1314_ = phi_2360_;
                                let _e1317_ = bitcast<u32>(_e1215_);
                                let _e1318_ = (_e1317_ & 2139095040u);
                                if (_e1318_ == 2139095040u) {
                                    phi_2376_ = true;
                                } else {
                                    phi_2376_ = (_e1318_ == 0u);
                                }
                                let _e1322_ = phi_2376_;
                                let _e1326_ = (_e1292_ < _e160_);
                                if _e1326_ {
                                    if _e1326_ {
                                    } else {
                                        phi_36252_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1292_] = vec4<f32>(bitcast<f32>(select(_e1293_, 0u, _e1298_)), bitcast<f32>(select(_e1301_, 0u, _e1306_)), bitcast<f32>(select(_e1309_, 0u, _e1314_)), bitcast<f32>(select(_e1317_, 0u, _e1322_)));
                                }
                                let _e1329_ = (_e1191_ + 2u);
                                let _e1330_ = bitcast<u32>(_e1201_);
                                let _e1331_ = (_e1330_ & 2139095040u);
                                if (_e1331_ == 2139095040u) {
                                    phi_2404_ = true;
                                } else {
                                    phi_2404_ = (_e1331_ == 0u);
                                }
                                let _e1335_ = phi_2404_;
                                let _e1338_ = bitcast<u32>(_e1203_);
                                let _e1339_ = (_e1338_ & 2139095040u);
                                if (_e1339_ == 2139095040u) {
                                    phi_2420_ = true;
                                } else {
                                    phi_2420_ = (_e1339_ == 0u);
                                }
                                let _e1343_ = phi_2420_;
                                let _e1346_ = bitcast<u32>(_e1205_);
                                let _e1347_ = (_e1346_ & 2139095040u);
                                if (_e1347_ == 2139095040u) {
                                    phi_2436_ = true;
                                } else {
                                    phi_2436_ = (_e1347_ == 0u);
                                }
                                let _e1351_ = phi_2436_;
                                let _e1354_ = bitcast<u32>(_e1207_);
                                let _e1355_ = (_e1354_ & 2139095040u);
                                if (_e1355_ == 2139095040u) {
                                    phi_2452_ = true;
                                } else {
                                    phi_2452_ = (_e1355_ == 0u);
                                }
                                let _e1359_ = phi_2452_;
                                let _e1363_ = (_e1329_ < _e160_);
                                if _e1363_ {
                                    if _e1363_ {
                                    } else {
                                        phi_36252_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1329_] = vec4<f32>(bitcast<f32>(select(_e1330_, 0u, _e1335_)), bitcast<f32>(select(_e1338_, 0u, _e1343_)), bitcast<f32>(select(_e1346_, 0u, _e1351_)), bitcast<f32>(select(_e1354_, 0u, _e1359_)));
                                }
                                let _e1366_ = (_e1191_ + 3u);
                                let _e1367_ = bitcast<u32>(_e1217_);
                                let _e1368_ = (_e1367_ & 2139095040u);
                                if (_e1368_ == 2139095040u) {
                                    phi_2480_ = true;
                                } else {
                                    phi_2480_ = (_e1368_ == 0u);
                                }
                                let _e1372_ = phi_2480_;
                                let _e1375_ = bitcast<u32>(_e1219_);
                                let _e1376_ = (_e1375_ & 2139095040u);
                                if (_e1376_ == 2139095040u) {
                                    phi_2496_ = true;
                                } else {
                                    phi_2496_ = (_e1376_ == 0u);
                                }
                                let _e1380_ = phi_2496_;
                                let _e1383_ = bitcast<u32>(_e1221_);
                                let _e1384_ = (_e1383_ & 2139095040u);
                                if (_e1384_ == 2139095040u) {
                                    phi_2512_ = true;
                                } else {
                                    phi_2512_ = (_e1384_ == 0u);
                                }
                                let _e1388_ = phi_2512_;
                                let _e1391_ = bitcast<u32>(_e1223_);
                                let _e1392_ = (_e1391_ & 2139095040u);
                                if (_e1392_ == 2139095040u) {
                                    phi_2528_ = true;
                                } else {
                                    phi_2528_ = (_e1392_ == 0u);
                                }
                                let _e1396_ = phi_2528_;
                                let _e1400_ = (_e1366_ < _e160_);
                                if _e1400_ {
                                    if _e1400_ {
                                    } else {
                                        phi_36252_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1366_] = vec4<f32>(bitcast<f32>(select(_e1367_, 0u, _e1372_)), bitcast<f32>(select(_e1375_, 0u, _e1380_)), bitcast<f32>(select(_e1383_, 0u, _e1388_)), bitcast<f32>(select(_e1391_, 0u, _e1396_)));
                                }
                                let _e1403_ = (_e1191_ + 4u);
                                let _e1404_ = bitcast<u32>(_e1225_);
                                let _e1405_ = (_e1404_ & 2139095040u);
                                if (_e1405_ == 2139095040u) {
                                    phi_2556_ = true;
                                } else {
                                    phi_2556_ = (_e1405_ == 0u);
                                }
                                let _e1409_ = phi_2556_;
                                let _e1412_ = bitcast<u32>(_e1227_);
                                let _e1413_ = (_e1412_ & 2139095040u);
                                if (_e1413_ == 2139095040u) {
                                    phi_2572_ = true;
                                } else {
                                    phi_2572_ = (_e1413_ == 0u);
                                }
                                let _e1417_ = phi_2572_;
                                let _e1420_ = bitcast<u32>(_e1229_);
                                let _e1421_ = (_e1420_ & 2139095040u);
                                if (_e1421_ == 2139095040u) {
                                    phi_2588_ = true;
                                } else {
                                    phi_2588_ = (_e1421_ == 0u);
                                }
                                let _e1425_ = phi_2588_;
                                let _e1428_ = bitcast<u32>(_e1231_);
                                let _e1429_ = (_e1428_ & 2139095040u);
                                if (_e1429_ == 2139095040u) {
                                    phi_2604_ = true;
                                } else {
                                    phi_2604_ = (_e1429_ == 0u);
                                }
                                let _e1433_ = phi_2604_;
                                let _e1437_ = (_e1403_ < _e160_);
                                if _e1437_ {
                                    if _e1437_ {
                                    } else {
                                        phi_36252_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1403_] = vec4<f32>(bitcast<f32>(select(_e1404_, 0u, _e1409_)), bitcast<f32>(select(_e1412_, 0u, _e1417_)), bitcast<f32>(select(_e1420_, 0u, _e1425_)), bitcast<f32>(select(_e1428_, 0u, _e1433_)));
                                }
                                let _e1440_ = (_e1191_ + 5u);
                                let _e1441_ = bitcast<u32>(_e1233_);
                                let _e1442_ = (_e1441_ & 2139095040u);
                                if (_e1442_ == 2139095040u) {
                                    phi_2632_ = true;
                                } else {
                                    phi_2632_ = (_e1442_ == 0u);
                                }
                                let _e1446_ = phi_2632_;
                                let _e1449_ = bitcast<u32>(_e1235_);
                                let _e1450_ = (_e1449_ & 2139095040u);
                                if (_e1450_ == 2139095040u) {
                                    phi_2648_ = true;
                                } else {
                                    phi_2648_ = (_e1450_ == 0u);
                                }
                                let _e1454_ = phi_2648_;
                                let _e1457_ = bitcast<u32>(_e1237_);
                                let _e1458_ = (_e1457_ & 2139095040u);
                                if (_e1458_ == 2139095040u) {
                                    phi_2664_ = true;
                                } else {
                                    phi_2664_ = (_e1458_ == 0u);
                                }
                                let _e1462_ = phi_2664_;
                                let _e1465_ = bitcast<u32>(_e1239_);
                                let _e1466_ = (_e1465_ & 2139095040u);
                                if (_e1466_ == 2139095040u) {
                                    phi_2680_ = true;
                                } else {
                                    phi_2680_ = (_e1466_ == 0u);
                                }
                                let _e1470_ = phi_2680_;
                                let _e1474_ = (_e1440_ < _e160_);
                                if _e1474_ {
                                    if _e1474_ {
                                    } else {
                                        phi_36252_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1440_] = vec4<f32>(bitcast<f32>(select(_e1441_, 0u, _e1446_)), bitcast<f32>(select(_e1449_, 0u, _e1454_)), bitcast<f32>(select(_e1457_, 0u, _e1462_)), bitcast<f32>(select(_e1465_, 0u, _e1470_)));
                                }
                                let _e1477_ = (_e1191_ + 6u);
                                let _e1478_ = bitcast<u32>(_e1241_);
                                let _e1479_ = (_e1478_ & 2139095040u);
                                if (_e1479_ == 2139095040u) {
                                    phi_2708_ = true;
                                } else {
                                    phi_2708_ = (_e1479_ == 0u);
                                }
                                let _e1483_ = phi_2708_;
                                let _e1486_ = bitcast<u32>(_e1243_);
                                let _e1487_ = (_e1486_ & 2139095040u);
                                if (_e1487_ == 2139095040u) {
                                    phi_2724_ = true;
                                } else {
                                    phi_2724_ = (_e1487_ == 0u);
                                }
                                let _e1491_ = phi_2724_;
                                let _e1494_ = bitcast<u32>(_e1245_);
                                let _e1495_ = (_e1494_ & 2139095040u);
                                if (_e1495_ == 2139095040u) {
                                    phi_2740_ = true;
                                } else {
                                    phi_2740_ = (_e1495_ == 0u);
                                }
                                let _e1499_ = phi_2740_;
                                let _e1502_ = bitcast<u32>(_e1247_);
                                let _e1503_ = (_e1502_ & 2139095040u);
                                if (_e1503_ == 2139095040u) {
                                    phi_2756_ = true;
                                } else {
                                    phi_2756_ = (_e1503_ == 0u);
                                }
                                let _e1507_ = phi_2756_;
                                let _e1511_ = (_e1477_ < _e160_);
                                if _e1511_ {
                                    if _e1511_ {
                                    } else {
                                        phi_36252_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1477_] = vec4<f32>(bitcast<f32>(select(_e1478_, 0u, _e1483_)), bitcast<f32>(select(_e1486_, 0u, _e1491_)), bitcast<f32>(select(_e1494_, 0u, _e1499_)), bitcast<f32>(select(_e1502_, 0u, _e1507_)));
                                }
                                let _e1514_ = (_e1191_ + 7u);
                                let _e1515_ = bitcast<u32>(_e1249_);
                                let _e1516_ = (_e1515_ & 2139095040u);
                                if (_e1516_ == 2139095040u) {
                                    phi_2784_ = true;
                                } else {
                                    phi_2784_ = (_e1516_ == 0u);
                                }
                                let _e1520_ = phi_2784_;
                                let _e1523_ = bitcast<u32>(_e1251_);
                                let _e1524_ = (_e1523_ & 2139095040u);
                                if (_e1524_ == 2139095040u) {
                                    phi_2800_ = true;
                                } else {
                                    phi_2800_ = (_e1524_ == 0u);
                                }
                                let _e1528_ = phi_2800_;
                                let _e1531_ = bitcast<u32>(_e1253_);
                                let _e1532_ = (_e1531_ & 2139095040u);
                                if (_e1532_ == 2139095040u) {
                                    phi_2816_ = true;
                                } else {
                                    phi_2816_ = (_e1532_ == 0u);
                                }
                                let _e1536_ = phi_2816_;
                                let _e1539_ = bitcast<u32>(_e1255_);
                                let _e1540_ = (_e1539_ & 2139095040u);
                                if (_e1540_ == 2139095040u) {
                                    phi_2832_ = true;
                                } else {
                                    phi_2832_ = (_e1540_ == 0u);
                                }
                                let _e1544_ = phi_2832_;
                                let _e1548_ = (_e1514_ < _e160_);
                                if _e1548_ {
                                    if _e1548_ {
                                    } else {
                                        phi_36252_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1514_] = vec4<f32>(bitcast<f32>(select(_e1515_, 0u, _e1520_)), bitcast<f32>(select(_e1523_, 0u, _e1528_)), bitcast<f32>(select(_e1531_, 0u, _e1536_)), bitcast<f32>(select(_e1539_, 0u, _e1544_)));
                                }
                            }
                        }
                        workgroupBarrier();
                        phi_33863_ = _e1161_;
                        phi_2851_ = type_15_(0u, 2u);
                        loop {
                            let _e1552_ = phi_33863_;
                            let _e1554_ = phi_2851_;
                            if (_e1554_.member < _e1554_.member_1_) {
                                phi_2852_ = type_15_((_e1554_.member + 1u), _e1554_.member_1_);
                                phi_2875_ = type_15_(1u, _e1554_.member);
                            } else {
                                phi_2852_ = _e1554_;
                                phi_2875_ = type_15_(0u, u32());
                            }
                            let _e1567_ = phi_2852_;
                            let _e1569_ = phi_2875_;
                            let _e1573_ = (bitcast<i32>(_e1569_.member) != 0i);
                            if _e1573_ {
                                let _e1575_ = ((_e1178_.member_1_ * 256u) + (_e1569_.member_1_ * 128u));
                                if (_e210_ > _e1575_) {
                                    phi_2890_ = (_e210_ - _e1575_);
                                } else {
                                    phi_2890_ = 0u;
                                }
                                let _e1579_ = phi_2890_;
                                let _e1581_ = select(_e1579_, 128u, (128u < _e1579_));
                                let _e1582_ = (_e1581_ > 0u);
                                if _e1582_ {
                                    let _e1583_ = (_e164_ < _e1581_);
                                    let _e1586_ = ((_e199_ * _e210_) + (_e1575_ + _e164_));
                                    let _e1587_ = (_e1586_ * 2u);
                                    if _e1583_ {
                                        let _e1588_ = (_e1587_ < _e156_);
                                        if _e1588_ {
                                            if _e1588_ {
                                            } else {
                                                phi_36246_ = true;
                                                break;
                                            }
                                            let _e1591_ = core_inputs.member[_e1587_];
                                            phi_2919_ = _e1591_;
                                        } else {
                                            phi_2919_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e1593_ = phi_2919_;
                                        let _e1594_ = (_e1587_ + 1u);
                                        let _e1595_ = (_e1594_ < _e156_);
                                        if _e1595_ {
                                            if _e1595_ {
                                            } else {
                                                phi_36246_ = true;
                                                break;
                                            }
                                            let _e1598_ = core_inputs.member[_e1594_];
                                            phi_2932_ = _e1598_;
                                        } else {
                                            phi_2932_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e1600_ = phi_2932_;
                                        phi_2933_ = _e1600_;
                                        phi_2934_ = _e1593_;
                                    } else {
                                        phi_2933_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        phi_2934_ = vec4<f32>(0f, 0f, 0f, 0f);
                                    }
                                    let _e1602_ = phi_2933_;
                                    let _e1604_ = phi_2934_;
                                    if _e207_ {
                                        if _e203_ {
                                            phi_34060_ = _e1552_;
                                            phi_2943_ = type_15_(0u, 4u);
                                            loop {
                                                let _e2264_ = phi_34060_;
                                                let _e2266_ = phi_2943_;
                                                if (_e2266_.member < _e2266_.member_1_) {
                                                    phi_2944_ = type_15_((_e2266_.member + 1u), _e2266_.member_1_);
                                                    phi_2967_ = type_15_(1u, _e2266_.member);
                                                } else {
                                                    phi_2944_ = _e2266_;
                                                    phi_2967_ = type_15_(0u, u32());
                                                }
                                                let _e2279_ = phi_2944_;
                                                let _e2281_ = phi_2967_;
                                                let _e2285_ = (bitcast<i32>(_e2281_.member) != 0i);
                                                if _e2285_ {
                                                    let _e2286_ = (_e2281_.member_1_ & 3u);
                                                    switch bitcast<i32>(_e2286_) {
                                                        case 0: {
                                                            phi_3080_ = _e327_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3080_ = _e327_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3080_ = _e327_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3080_ = _e327_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2293_ = phi_3080_;
                                                    switch bitcast<i32>(_e2286_) {
                                                        case 0: {
                                                            phi_3097_ = _e341_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3097_ = _e341_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3097_ = _e341_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3097_ = _e341_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2300_ = phi_3097_;
                                                    switch bitcast<i32>(_e2286_) {
                                                        case 0: {
                                                            phi_3108_ = _e348_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3108_ = _e348_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3108_ = _e348_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3108_ = _e348_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2307_ = phi_3108_;
                                                    switch bitcast<i32>(_e2286_) {
                                                        case 0: {
                                                            phi_3119_ = _e355_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3119_ = _e355_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3119_ = _e355_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3119_ = _e355_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2314_ = phi_3119_;
                                                    switch bitcast<i32>(_e2286_) {
                                                        case 0: {
                                                            phi_3130_ = _e362_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3130_ = _e362_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3130_ = _e362_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3130_ = _e362_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2321_ = phi_3130_;
                                                    switch bitcast<i32>(_e2286_) {
                                                        case 0: {
                                                            phi_3141_ = _e369_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3141_ = _e369_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3141_ = _e369_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3141_ = _e369_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2328_ = phi_3141_;
                                                    switch bitcast<i32>(_e2286_) {
                                                        case 0: {
                                                            phi_3356_ = _e418_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3356_ = _e418_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3356_ = _e418_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3356_ = _e418_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2335_ = phi_3356_;
                                                    switch bitcast<i32>(_e2286_) {
                                                        case 0: {
                                                            phi_3431_ = _e469_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3431_ = _e469_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3431_ = _e469_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3431_ = _e469_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2342_ = phi_3431_;
                                                    if _e1583_ {
                                                        let _e2344_ = select(_e1604_, _e1602_, vec4(_e167_));
                                                        switch bitcast<i32>(_e2286_) {
                                                            case 0: {
                                                                phi_3449_ = _e2344_.x;
                                                                break;
                                                            }
                                                            case 1: {
                                                                phi_3449_ = _e2344_.y;
                                                                break;
                                                            }
                                                            case 2: {
                                                                phi_3449_ = _e2344_.z;
                                                                break;
                                                            }
                                                            default: {
                                                                phi_3449_ = _e2344_.w;
                                                                break;
                                                            }
                                                        }
                                                        let _e2351_ = phi_3449_;
                                                        let _e2352_ = bitcast<u32>(_e2351_);
                                                        let _e2353_ = (_e2352_ & 2139095040u);
                                                        if (_e2353_ == 2139095040u) {
                                                            phi_3460_ = true;
                                                        } else {
                                                            phi_3460_ = (_e2353_ == 0u);
                                                        }
                                                        let _e2357_ = phi_3460_;
                                                        phi_3466_ = bitcast<f32>(select(_e2352_, 0u, _e2357_));
                                                    } else {
                                                        phi_3466_ = 0f;
                                                    }
                                                    let _e2361_ = phi_3466_;
                                                    if _e167_ {
                                                        phi_3475_ = (8u + _e2281_.member_1_);
                                                    } else {
                                                        phi_3475_ = _e2281_.member_1_;
                                                    }
                                                    let _e2364_ = phi_3475_;
                                                    if (_e2364_ < 256u) {
                                                    } else {
                                                        phi_34247_ = true;
                                                        break;
                                                    }
                                                    let _e2367_ = aux_scratch[_e2364_];
                                                    let _e2368_ = (_e2364_ + 4u);
                                                    if (_e2368_ < 256u) {
                                                    } else {
                                                        phi_34247_ = true;
                                                        break;
                                                    }
                                                    let _e2371_ = aux_scratch[_e2368_];
                                                    let _e2372_ = array<f32, 2>(_e2367_, _e2371_);
                                                    phi_34048_ = _e2264_;
                                                    phi_3494_ = type_15_(0u, 6u);
                                                    phi_3497_ = _e2372_;
                                                    phi_3499_ = _e2371_;
                                                    phi_3501_ = _e2367_;
                                                    phi_3503_ = _e2372_;
                                                    phi_3505_ = false;
                                                    loop {
                                                        let _e2374_ = phi_34048_;
                                                        let _e2376_ = phi_3494_;
                                                        let _e2378_ = phi_3497_;
                                                        let _e2380_ = phi_3499_;
                                                        let _e2382_ = phi_3501_;
                                                        let _e2384_ = phi_3503_;
                                                        let _e2386_ = phi_3505_;
                                                        local_24_ = _e2386_;
                                                        local_29_ = _e2382_;
                                                        local_30_ = _e2382_;
                                                        local_31_ = _e2380_;
                                                        local_32_ = _e2382_;
                                                        local_33_ = _e2382_;
                                                        local_34_ = _e2380_;
                                                        local_35_ = _e2384_;
                                                        local_36_ = _e2384_;
                                                        local_37_ = _e2384_;
                                                        local_38_ = _e2384_;
                                                        if (_e2376_.member < _e2376_.member_1_) {
                                                            phi_3495_ = type_15_((_e2376_.member + 1u), _e2376_.member_1_);
                                                            phi_3528_ = type_15_(1u, _e2376_.member);
                                                        } else {
                                                            phi_3495_ = _e2376_;
                                                            phi_3528_ = type_15_(0u, u32());
                                                        }
                                                        let _e2399_ = phi_3495_;
                                                        let _e2401_ = phi_3528_;
                                                        let _e2405_ = (bitcast<i32>(_e2401_.member) != 0i);
                                                        if _e2405_ {
                                                            if _e2386_ {
                                                                phi_34159_ = _e2374_;
                                                                phi_4625_ = _e2378_;
                                                                phi_4626_ = _e2380_;
                                                                phi_4627_ = _e2382_;
                                                                phi_4628_ = _e2384_;
                                                                phi_4629_ = _e2386_;
                                                            } else {
                                                                let _e2406_ = (_e2335_ * 0.5f);
                                                                if (_e2406_ != _e2406_) {
                                                                    phi_23660_ = true;
                                                                } else {
                                                                    phi_23660_ = (0.00001f >= _e2406_);
                                                                }
                                                                let _e2410_ = phi_23660_;
                                                                let _e2411_ = select(_e2406_, 0.00001f, _e2410_);
                                                                if (_e2293_ != _e2293_) {
                                                                    phi_23675_ = true;
                                                                } else {
                                                                    phi_23675_ = (0f >= _e2293_);
                                                                }
                                                                let _e2415_ = phi_23675_;
                                                                if (_e2342_ != _e2342_) {
                                                                    phi_23690_ = true;
                                                                } else {
                                                                    phi_23690_ = (0f >= _e2342_);
                                                                }
                                                                let _e2420_ = phi_23690_;
                                                                let _e2422_ = (_e2382_ * _e2382_);
                                                                if (_e2422_ != _e2422_) {
                                                                    phi_23705_ = true;
                                                                } else {
                                                                    phi_23705_ = (64f <= _e2422_);
                                                                }
                                                                let _e2426_ = phi_23705_;
                                                                let _e2432_ = (1f + (_e2411_ * (_e2411_ + (select(_e2293_, 0f, _e2415_) + (select(_e2342_, 0f, _e2420_) * select(_e2422_, 64f, _e2426_))))));
                                                                let _e2435_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2432_)));
                                                                let _e2439_ = (_e2435_ + (_e2435_ * (1f - (_e2432_ * _e2435_))));
                                                                let _e2443_ = (_e2439_ + (_e2439_ * (1f - (_e2432_ * _e2439_))));
                                                                let _e2444_ = (_e2411_ * _e2443_);
                                                                let _e2445_ = (_e2411_ * _e2444_);
                                                                if _e1583_ {
                                                                    let _e2446_ = (_e2444_ + _e2444_);
                                                                    let _e2447_ = (_e2445_ + _e2445_);
                                                                    phi_3576_ = type_31_(((_e2443_ + _e2443_) - 1f), -(_e2446_), _e2446_, (1f - _e2447_), (_e2446_ * _e2361_), (_e2447_ * _e2361_));
                                                                } else {
                                                                    phi_3576_ = type_31_(1f, 0f, 0f, 1f, 0f, 0f);
                                                                }
                                                                let _e2456_ = phi_3576_;
                                                                let _e2457_ = select(0u, 768u, _e167_);
                                                                let _e2458_ = (_e164_ & 127u);
                                                                let _e2460_ = (_e2457_ + _e2458_);
                                                                let _e2461_ = (_e2460_ < 1536u);
                                                                if _e2461_ {
                                                                } else {
                                                                    phi_34157_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2460_] = _e2456_.member;
                                                                let _e2464_ = (_e2457_ + 128u);
                                                                let _e2465_ = (_e2464_ + _e2458_);
                                                                let _e2466_ = (_e2465_ < 1536u);
                                                                if _e2466_ {
                                                                } else {
                                                                    phi_34157_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2465_] = _e2456_.member_1_;
                                                                let _e2469_ = (_e2457_ + 256u);
                                                                let _e2470_ = (_e2469_ + _e2458_);
                                                                let _e2471_ = (_e2470_ < 1536u);
                                                                if _e2471_ {
                                                                } else {
                                                                    phi_34157_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2470_] = _e2456_.member_2_;
                                                                let _e2474_ = (_e2457_ + 384u);
                                                                let _e2475_ = (_e2474_ + _e2458_);
                                                                let _e2476_ = (_e2475_ < 1536u);
                                                                if _e2476_ {
                                                                } else {
                                                                    phi_34157_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2475_] = _e2456_.member_3_;
                                                                let _e2479_ = (_e2457_ + 512u);
                                                                let _e2480_ = (_e2479_ + _e2458_);
                                                                let _e2481_ = (_e2480_ < 1536u);
                                                                if _e2481_ {
                                                                } else {
                                                                    phi_34157_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2480_] = _e2456_.member_4_;
                                                                let _e2484_ = (_e2457_ + 640u);
                                                                let _e2485_ = (_e2484_ + _e2458_);
                                                                let _e2486_ = (_e2485_ < 1536u);
                                                                if _e2486_ {
                                                                } else {
                                                                    phi_34157_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2485_] = _e2456_.member_5_;
                                                                workgroupBarrier();
                                                                phi_3639_ = 1u;
                                                                phi_3642_ = _e2456_;
                                                                loop {
                                                                    let _e2489_ = phi_3639_;
                                                                    let _e2491_ = phi_3642_;
                                                                    let _e2492_ = (_e2489_ < 128u);
                                                                    if _e2492_ {
                                                                        if (_e164_ >= _e2489_) {
                                                                            let _e2495_ = ((_e164_ - _e2489_) & 127u);
                                                                            let _e2496_ = (_e2457_ + _e2495_);
                                                                            if (_e2496_ < 1536u) {
                                                                            } else {
                                                                                phi_34034_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2499_ = scan_scratch[_e2496_];
                                                                            let _e2500_ = (_e2464_ + _e2495_);
                                                                            if (_e2500_ < 1536u) {
                                                                            } else {
                                                                                phi_34034_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2503_ = scan_scratch[_e2500_];
                                                                            let _e2504_ = (_e2469_ + _e2495_);
                                                                            if (_e2504_ < 1536u) {
                                                                            } else {
                                                                                phi_34034_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2507_ = scan_scratch[_e2504_];
                                                                            let _e2508_ = (_e2474_ + _e2495_);
                                                                            if (_e2508_ < 1536u) {
                                                                            } else {
                                                                                phi_34034_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2511_ = scan_scratch[_e2508_];
                                                                            let _e2512_ = (_e2479_ + _e2495_);
                                                                            if (_e2512_ < 1536u) {
                                                                            } else {
                                                                                phi_34034_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2515_ = scan_scratch[_e2512_];
                                                                            let _e2516_ = (_e2484_ + _e2495_);
                                                                            if (_e2516_ < 1536u) {
                                                                            } else {
                                                                                phi_34034_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2519_ = scan_scratch[_e2516_];
                                                                            phi_3744_ = type_31_(((_e2491_.member * _e2499_) + (_e2491_.member_1_ * _e2507_)), ((_e2491_.member * _e2503_) + (_e2491_.member_1_ * _e2511_)), ((_e2491_.member_2_ * _e2499_) + (_e2491_.member_3_ * _e2507_)), ((_e2491_.member_2_ * _e2503_) + (_e2491_.member_3_ * _e2511_)), (((_e2491_.member * _e2515_) + (_e2491_.member_1_ * _e2519_)) + _e2491_.member_4_), (((_e2491_.member_2_ * _e2515_) + (_e2491_.member_3_ * _e2519_)) + _e2491_.member_5_));
                                                                        } else {
                                                                            phi_3744_ = _e2491_;
                                                                        }
                                                                        let _e2548_ = phi_3744_;
                                                                        workgroupBarrier();
                                                                        if _e2461_ {
                                                                        } else {
                                                                            phi_34034_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2460_] = _e2548_.member;
                                                                        if _e2466_ {
                                                                        } else {
                                                                            phi_34034_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2465_] = _e2548_.member_1_;
                                                                        if _e2471_ {
                                                                        } else {
                                                                            phi_34034_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2470_] = _e2548_.member_2_;
                                                                        if _e2476_ {
                                                                        } else {
                                                                            phi_34034_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2475_] = _e2548_.member_3_;
                                                                        if _e2481_ {
                                                                        } else {
                                                                            phi_34034_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2480_] = _e2548_.member_4_;
                                                                        if _e2486_ {
                                                                        } else {
                                                                            phi_34034_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2485_] = _e2548_.member_5_;
                                                                        workgroupBarrier();
                                                                        phi_3640_ = (_e2489_ << bitcast<u32>(1i));
                                                                        phi_3643_ = _e2548_;
                                                                    } else {
                                                                        phi_3640_ = u32();
                                                                        phi_3643_ = type_31_();
                                                                    }
                                                                    let _e2558_ = phi_3640_;
                                                                    let _e2560_ = phi_3643_;
                                                                    phi_3639_ = _e2558_;
                                                                    phi_3642_ = _e2560_;
                                                                    phi_34034_ = _e2374_;
                                                                    if !(_e2492_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e2563_ = phi_34034_;
                                                                phi_34157_ = _e2563_;
                                                                if _e2563_ {
                                                                    break;
                                                                }
                                                                if (_e164_ == 0u) {
                                                                    phi_3868_ = _e2372_;
                                                                } else {
                                                                    let _e2566_ = ((_e164_ - 1u) & 127u);
                                                                    let _e2567_ = (_e2457_ + _e2566_);
                                                                    if (_e2567_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2570_ = scan_scratch[_e2567_];
                                                                    let _e2571_ = (_e2464_ + _e2566_);
                                                                    if (_e2571_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2574_ = scan_scratch[_e2571_];
                                                                    let _e2575_ = (_e2469_ + _e2566_);
                                                                    if (_e2575_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2578_ = scan_scratch[_e2575_];
                                                                    let _e2579_ = (_e2474_ + _e2566_);
                                                                    if (_e2579_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2582_ = scan_scratch[_e2579_];
                                                                    let _e2583_ = (_e2479_ + _e2566_);
                                                                    if (_e2583_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2586_ = scan_scratch[_e2583_];
                                                                    let _e2587_ = (_e2484_ + _e2566_);
                                                                    if (_e2587_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2590_ = scan_scratch[_e2587_];
                                                                    phi_3868_ = array<f32, 2>((((_e2570_ * _e2367_) + (_e2574_ * _e2371_)) + _e2586_), (((_e2578_ * _e2367_) + (_e2582_ * _e2371_)) + _e2590_));
                                                                }
                                                                let _e2601_ = phi_3868_;
                                                                if _e1582_ {
                                                                    if (_e1581_ <= 128u) {
                                                                        phi_3880_ = (_e1581_ - 1u);
                                                                    } else {
                                                                        phi_3880_ = 127u;
                                                                    }
                                                                    let _e2607_ = phi_3880_;
                                                                    let _e2608_ = (_e2607_ & 127u);
                                                                    let _e2609_ = (_e2457_ + _e2608_);
                                                                    if (_e2609_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2612_ = scan_scratch[_e2609_];
                                                                    let _e2613_ = (_e2464_ + _e2608_);
                                                                    if (_e2613_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2616_ = scan_scratch[_e2613_];
                                                                    let _e2617_ = (_e2469_ + _e2608_);
                                                                    if (_e2617_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2620_ = scan_scratch[_e2617_];
                                                                    let _e2621_ = (_e2474_ + _e2608_);
                                                                    if (_e2621_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2624_ = scan_scratch[_e2621_];
                                                                    let _e2625_ = (_e2479_ + _e2608_);
                                                                    if (_e2625_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2628_ = scan_scratch[_e2625_];
                                                                    let _e2629_ = (_e2484_ + _e2608_);
                                                                    if (_e2629_ < 1536u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2632_ = scan_scratch[_e2629_];
                                                                    phi_3944_ = array<f32, 2>((((_e2612_ * _e2367_) + (_e2616_ * _e2371_)) + _e2628_), (((_e2620_ * _e2367_) + (_e2624_ * _e2371_)) + _e2632_));
                                                                } else {
                                                                    phi_3944_ = _e2372_;
                                                                }
                                                                let _e2643_ = phi_3944_;
                                                                workgroupBarrier();
                                                                if _e1583_ {
                                                                    if ((bitcast<u32>(_e2601_[0]) & 2139095040u) != 2139095040u) {
                                                                        let _e2649_ = ((bitcast<u32>(_e2382_) & 2139095040u) != 2139095040u);
                                                                        if _e2649_ {
                                                                            let _e2650_ = abs(_e2601_[0]);
                                                                            let _e2651_ = abs(_e2382_);
                                                                            let _e2653_ = select(_e2651_, _e2650_, (_e2650_ > _e2651_));
                                                                            let _e2655_ = select(1f, _e2653_, (_e2653_ > 1f));
                                                                            if ((bitcast<u32>(_e2655_) & 2139095040u) != 2139095040u) {
                                                                                let _e2661_ = (_e2655_ <= 0f);
                                                                                if _e2661_ {
                                                                                    phi_4066_ = f32();
                                                                                } else {
                                                                                    if (_e2655_ < 0.00001f) {
                                                                                        let _e2693_ = (select(0.000000000000000000000001f, _e2655_, (_e2655_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                        if (_e2693_ != _e2693_) {
                                                                                            phi_23780_ = true;
                                                                                        } else {
                                                                                            phi_23780_ = (0.000001f >= _e2693_);
                                                                                        }
                                                                                        let _e2697_ = phi_23780_;
                                                                                        let _e2698_ = select(_e2693_, 0.000001f, _e2697_);
                                                                                        let _e2701_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2698_)));
                                                                                        let _e2705_ = (_e2701_ + (_e2701_ * (1f - (_e2698_ * _e2701_))));
                                                                                        let _e2709_ = (_e2705_ + (_e2705_ * (1f - (_e2698_ * _e2705_))));
                                                                                        let _e2715_ = bitcast<u32>(((_e2709_ + (_e2709_ * (1f - (_e2698_ * _e2709_)))) * 18446744000000000000f));
                                                                                        let _e2716_ = (_e2715_ & 2139095040u);
                                                                                        if (_e2716_ == 2139095040u) {
                                                                                            phi_4024_ = true;
                                                                                        } else {
                                                                                            phi_4024_ = (_e2716_ == 0u);
                                                                                        }
                                                                                        let _e2720_ = phi_4024_;
                                                                                        phi_4065_ = bitcast<f32>(select(_e2715_, 0u, _e2720_));
                                                                                    } else {
                                                                                        if (_e2655_ != _e2655_) {
                                                                                            phi_23720_ = true;
                                                                                        } else {
                                                                                            phi_23720_ = (0.000001f >= _e2655_);
                                                                                        }
                                                                                        let _e2666_ = phi_23720_;
                                                                                        let _e2667_ = select(_e2655_, 0.000001f, _e2666_);
                                                                                        let _e2670_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2667_)));
                                                                                        let _e2674_ = (_e2670_ + (_e2670_ * (1f - (_e2667_ * _e2670_))));
                                                                                        let _e2678_ = (_e2674_ + (_e2674_ * (1f - (_e2667_ * _e2674_))));
                                                                                        let _e2683_ = bitcast<u32>((_e2678_ + (_e2678_ * (1f - (_e2667_ * _e2678_)))));
                                                                                        let _e2684_ = (_e2683_ & 2139095040u);
                                                                                        if (_e2684_ == 2139095040u) {
                                                                                            phi_4059_ = true;
                                                                                        } else {
                                                                                            phi_4059_ = (_e2684_ == 0u);
                                                                                        }
                                                                                        let _e2688_ = phi_4059_;
                                                                                        phi_4065_ = bitcast<f32>(select(_e2683_, 0u, _e2688_));
                                                                                    }
                                                                                    let _e2724_ = phi_4065_;
                                                                                    phi_4066_ = _e2724_;
                                                                                }
                                                                                let _e2726_ = phi_4066_;
                                                                                phi_4067_ = _e2726_;
                                                                                phi_4068_ = _e2661_;
                                                                            } else {
                                                                                phi_4067_ = f32();
                                                                                phi_4068_ = true;
                                                                            }
                                                                            let _e2728_ = phi_4067_;
                                                                            let _e2730_ = phi_4068_;
                                                                            phi_4074_ = (abs((_e2601_[0] - _e2382_)) * select(_e2728_, 0f, _e2730_));
                                                                        } else {
                                                                            phi_4074_ = f32();
                                                                        }
                                                                        let _e2734_ = phi_4074_;
                                                                        phi_4076_ = _e2734_;
                                                                        phi_4077_ = select(true, false, _e2649_);
                                                                    } else {
                                                                        phi_4076_ = f32();
                                                                        phi_4077_ = true;
                                                                    }
                                                                    let _e2737_ = phi_4076_;
                                                                    let _e2739_ = phi_4077_;
                                                                    let _e2740_ = select(_e2737_, 1000000000f, _e2739_);
                                                                    if ((bitcast<u32>(_e2601_[1]) & 2139095040u) != 2139095040u) {
                                                                        let _e2746_ = ((bitcast<u32>(_e2380_) & 2139095040u) != 2139095040u);
                                                                        if _e2746_ {
                                                                            let _e2747_ = abs(_e2601_[1]);
                                                                            let _e2748_ = abs(_e2380_);
                                                                            let _e2750_ = select(_e2748_, _e2747_, (_e2747_ > _e2748_));
                                                                            let _e2752_ = select(1f, _e2750_, (_e2750_ > 1f));
                                                                            if ((bitcast<u32>(_e2752_) & 2139095040u) != 2139095040u) {
                                                                                let _e2758_ = (_e2752_ <= 0f);
                                                                                if _e2758_ {
                                                                                    phi_4200_ = f32();
                                                                                } else {
                                                                                    if (_e2752_ < 0.00001f) {
                                                                                        let _e2790_ = (select(0.000000000000000000000001f, _e2752_, (_e2752_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                        if (_e2790_ != _e2790_) {
                                                                                            phi_23900_ = true;
                                                                                        } else {
                                                                                            phi_23900_ = (0.000001f >= _e2790_);
                                                                                        }
                                                                                        let _e2794_ = phi_23900_;
                                                                                        let _e2795_ = select(_e2790_, 0.000001f, _e2794_);
                                                                                        let _e2798_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2795_)));
                                                                                        let _e2802_ = (_e2798_ + (_e2798_ * (1f - (_e2795_ * _e2798_))));
                                                                                        let _e2806_ = (_e2802_ + (_e2802_ * (1f - (_e2795_ * _e2802_))));
                                                                                        let _e2812_ = bitcast<u32>(((_e2806_ + (_e2806_ * (1f - (_e2795_ * _e2806_)))) * 18446744000000000000f));
                                                                                        let _e2813_ = (_e2812_ & 2139095040u);
                                                                                        if (_e2813_ == 2139095040u) {
                                                                                            phi_4158_ = true;
                                                                                        } else {
                                                                                            phi_4158_ = (_e2813_ == 0u);
                                                                                        }
                                                                                        let _e2817_ = phi_4158_;
                                                                                        phi_4199_ = bitcast<f32>(select(_e2812_, 0u, _e2817_));
                                                                                    } else {
                                                                                        if (_e2752_ != _e2752_) {
                                                                                            phi_23840_ = true;
                                                                                        } else {
                                                                                            phi_23840_ = (0.000001f >= _e2752_);
                                                                                        }
                                                                                        let _e2763_ = phi_23840_;
                                                                                        let _e2764_ = select(_e2752_, 0.000001f, _e2763_);
                                                                                        let _e2767_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2764_)));
                                                                                        let _e2771_ = (_e2767_ + (_e2767_ * (1f - (_e2764_ * _e2767_))));
                                                                                        let _e2775_ = (_e2771_ + (_e2771_ * (1f - (_e2764_ * _e2771_))));
                                                                                        let _e2780_ = bitcast<u32>((_e2775_ + (_e2775_ * (1f - (_e2764_ * _e2775_)))));
                                                                                        let _e2781_ = (_e2780_ & 2139095040u);
                                                                                        if (_e2781_ == 2139095040u) {
                                                                                            phi_4193_ = true;
                                                                                        } else {
                                                                                            phi_4193_ = (_e2781_ == 0u);
                                                                                        }
                                                                                        let _e2785_ = phi_4193_;
                                                                                        phi_4199_ = bitcast<f32>(select(_e2780_, 0u, _e2785_));
                                                                                    }
                                                                                    let _e2821_ = phi_4199_;
                                                                                    phi_4200_ = _e2821_;
                                                                                }
                                                                                let _e2823_ = phi_4200_;
                                                                                phi_4201_ = _e2823_;
                                                                                phi_4202_ = _e2758_;
                                                                            } else {
                                                                                phi_4201_ = f32();
                                                                                phi_4202_ = true;
                                                                            }
                                                                            let _e2825_ = phi_4201_;
                                                                            let _e2827_ = phi_4202_;
                                                                            phi_4208_ = (abs((_e2601_[1] - _e2380_)) * select(_e2825_, 0f, _e2827_));
                                                                        } else {
                                                                            phi_4208_ = f32();
                                                                        }
                                                                        let _e2831_ = phi_4208_;
                                                                        phi_4210_ = _e2831_;
                                                                        phi_4211_ = select(true, false, _e2746_);
                                                                    } else {
                                                                        phi_4210_ = f32();
                                                                        phi_4211_ = true;
                                                                    }
                                                                    let _e2834_ = phi_4210_;
                                                                    let _e2836_ = phi_4211_;
                                                                    let _e2837_ = select(_e2834_, 1000000000f, _e2836_);
                                                                    let _e2839_ = select(_e2740_, _e2837_, (_e2837_ > _e2740_));
                                                                    if (_e2401_.member_1_ > 0u) {
                                                                        let _e2842_ = ((_e164_ + 1u) == _e1581_);
                                                                        if _e2842_ {
                                                                            if ((bitcast<u32>(_e2643_[0]) & 2139095040u) != 2139095040u) {
                                                                                let _e2850_ = ((bitcast<u32>(_e2378_[0]) & 2139095040u) != 2139095040u);
                                                                                if _e2850_ {
                                                                                    let _e2851_ = abs(_e2643_[0]);
                                                                                    let _e2852_ = abs(_e2378_[0]);
                                                                                    let _e2854_ = select(_e2852_, _e2851_, (_e2851_ > _e2852_));
                                                                                    let _e2856_ = select(1f, _e2854_, (_e2854_ > 1f));
                                                                                    if ((bitcast<u32>(_e2856_) & 2139095040u) != 2139095040u) {
                                                                                        let _e2862_ = (_e2856_ <= 0f);
                                                                                        if _e2862_ {
                                                                                            phi_4350_ = f32();
                                                                                        } else {
                                                                                            if (_e2856_ < 0.00001f) {
                                                                                                let _e2894_ = (select(0.000000000000000000000001f, _e2856_, (_e2856_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                                if (_e2894_ != _e2894_) {
                                                                                                    phi_24020_ = true;
                                                                                                } else {
                                                                                                    phi_24020_ = (0.000001f >= _e2894_);
                                                                                                }
                                                                                                let _e2898_ = phi_24020_;
                                                                                                let _e2899_ = select(_e2894_, 0.000001f, _e2898_);
                                                                                                let _e2902_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2899_)));
                                                                                                let _e2906_ = (_e2902_ + (_e2902_ * (1f - (_e2899_ * _e2902_))));
                                                                                                let _e2910_ = (_e2906_ + (_e2906_ * (1f - (_e2899_ * _e2906_))));
                                                                                                let _e2916_ = bitcast<u32>(((_e2910_ + (_e2910_ * (1f - (_e2899_ * _e2910_)))) * 18446744000000000000f));
                                                                                                let _e2917_ = (_e2916_ & 2139095040u);
                                                                                                if (_e2917_ == 2139095040u) {
                                                                                                    phi_4308_ = true;
                                                                                                } else {
                                                                                                    phi_4308_ = (_e2917_ == 0u);
                                                                                                }
                                                                                                let _e2921_ = phi_4308_;
                                                                                                phi_4349_ = bitcast<f32>(select(_e2916_, 0u, _e2921_));
                                                                                            } else {
                                                                                                if (_e2856_ != _e2856_) {
                                                                                                    phi_23960_ = true;
                                                                                                } else {
                                                                                                    phi_23960_ = (0.000001f >= _e2856_);
                                                                                                }
                                                                                                let _e2867_ = phi_23960_;
                                                                                                let _e2868_ = select(_e2856_, 0.000001f, _e2867_);
                                                                                                let _e2871_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2868_)));
                                                                                                let _e2875_ = (_e2871_ + (_e2871_ * (1f - (_e2868_ * _e2871_))));
                                                                                                let _e2879_ = (_e2875_ + (_e2875_ * (1f - (_e2868_ * _e2875_))));
                                                                                                let _e2884_ = bitcast<u32>((_e2879_ + (_e2879_ * (1f - (_e2868_ * _e2879_)))));
                                                                                                let _e2885_ = (_e2884_ & 2139095040u);
                                                                                                if (_e2885_ == 2139095040u) {
                                                                                                    phi_4343_ = true;
                                                                                                } else {
                                                                                                    phi_4343_ = (_e2885_ == 0u);
                                                                                                }
                                                                                                let _e2889_ = phi_4343_;
                                                                                                phi_4349_ = bitcast<f32>(select(_e2884_, 0u, _e2889_));
                                                                                            }
                                                                                            let _e2925_ = phi_4349_;
                                                                                            phi_4350_ = _e2925_;
                                                                                        }
                                                                                        let _e2927_ = phi_4350_;
                                                                                        phi_4351_ = _e2927_;
                                                                                        phi_4352_ = _e2862_;
                                                                                    } else {
                                                                                        phi_4351_ = f32();
                                                                                        phi_4352_ = true;
                                                                                    }
                                                                                    let _e2929_ = phi_4351_;
                                                                                    let _e2931_ = phi_4352_;
                                                                                    phi_4358_ = (abs((_e2643_[0] - _e2378_[0])) * select(_e2929_, 0f, _e2931_));
                                                                                } else {
                                                                                    phi_4358_ = f32();
                                                                                }
                                                                                let _e2935_ = phi_4358_;
                                                                                phi_4360_ = _e2935_;
                                                                                phi_4361_ = select(true, false, _e2850_);
                                                                            } else {
                                                                                phi_4360_ = f32();
                                                                                phi_4361_ = true;
                                                                            }
                                                                            let _e2938_ = phi_4360_;
                                                                            let _e2940_ = phi_4361_;
                                                                            let _e2941_ = select(_e2938_, 1000000000f, _e2940_);
                                                                            if ((bitcast<u32>(_e2643_[1]) & 2139095040u) != 2139095040u) {
                                                                                let _e2949_ = ((bitcast<u32>(_e2378_[1]) & 2139095040u) != 2139095040u);
                                                                                if _e2949_ {
                                                                                    let _e2950_ = abs(_e2643_[1]);
                                                                                    let _e2951_ = abs(_e2378_[1]);
                                                                                    let _e2953_ = select(_e2951_, _e2950_, (_e2950_ > _e2951_));
                                                                                    let _e2955_ = select(1f, _e2953_, (_e2953_ > 1f));
                                                                                    if ((bitcast<u32>(_e2955_) & 2139095040u) != 2139095040u) {
                                                                                        let _e2961_ = (_e2955_ <= 0f);
                                                                                        if _e2961_ {
                                                                                            phi_4486_ = f32();
                                                                                        } else {
                                                                                            if (_e2955_ < 0.00001f) {
                                                                                                let _e2993_ = (select(0.000000000000000000000001f, _e2955_, (_e2955_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                                if (_e2993_ != _e2993_) {
                                                                                                    phi_24140_ = true;
                                                                                                } else {
                                                                                                    phi_24140_ = (0.000001f >= _e2993_);
                                                                                                }
                                                                                                let _e2997_ = phi_24140_;
                                                                                                let _e2998_ = select(_e2993_, 0.000001f, _e2997_);
                                                                                                let _e3001_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2998_)));
                                                                                                let _e3005_ = (_e3001_ + (_e3001_ * (1f - (_e2998_ * _e3001_))));
                                                                                                let _e3009_ = (_e3005_ + (_e3005_ * (1f - (_e2998_ * _e3005_))));
                                                                                                let _e3015_ = bitcast<u32>(((_e3009_ + (_e3009_ * (1f - (_e2998_ * _e3009_)))) * 18446744000000000000f));
                                                                                                let _e3016_ = (_e3015_ & 2139095040u);
                                                                                                if (_e3016_ == 2139095040u) {
                                                                                                    phi_4444_ = true;
                                                                                                } else {
                                                                                                    phi_4444_ = (_e3016_ == 0u);
                                                                                                }
                                                                                                let _e3020_ = phi_4444_;
                                                                                                phi_4485_ = bitcast<f32>(select(_e3015_, 0u, _e3020_));
                                                                                            } else {
                                                                                                if (_e2955_ != _e2955_) {
                                                                                                    phi_24080_ = true;
                                                                                                } else {
                                                                                                    phi_24080_ = (0.000001f >= _e2955_);
                                                                                                }
                                                                                                let _e2966_ = phi_24080_;
                                                                                                let _e2967_ = select(_e2955_, 0.000001f, _e2966_);
                                                                                                let _e2970_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2967_)));
                                                                                                let _e2974_ = (_e2970_ + (_e2970_ * (1f - (_e2967_ * _e2970_))));
                                                                                                let _e2978_ = (_e2974_ + (_e2974_ * (1f - (_e2967_ * _e2974_))));
                                                                                                let _e2983_ = bitcast<u32>((_e2978_ + (_e2978_ * (1f - (_e2967_ * _e2978_)))));
                                                                                                let _e2984_ = (_e2983_ & 2139095040u);
                                                                                                if (_e2984_ == 2139095040u) {
                                                                                                    phi_4479_ = true;
                                                                                                } else {
                                                                                                    phi_4479_ = (_e2984_ == 0u);
                                                                                                }
                                                                                                let _e2988_ = phi_4479_;
                                                                                                phi_4485_ = bitcast<f32>(select(_e2983_, 0u, _e2988_));
                                                                                            }
                                                                                            let _e3024_ = phi_4485_;
                                                                                            phi_4486_ = _e3024_;
                                                                                        }
                                                                                        let _e3026_ = phi_4486_;
                                                                                        phi_4487_ = _e3026_;
                                                                                        phi_4488_ = _e2961_;
                                                                                    } else {
                                                                                        phi_4487_ = f32();
                                                                                        phi_4488_ = true;
                                                                                    }
                                                                                    let _e3028_ = phi_4487_;
                                                                                    let _e3030_ = phi_4488_;
                                                                                    phi_4494_ = (abs((_e2643_[1] - _e2378_[1])) * select(_e3028_, 0f, _e3030_));
                                                                                } else {
                                                                                    phi_4494_ = f32();
                                                                                }
                                                                                let _e3034_ = phi_4494_;
                                                                                phi_4496_ = _e3034_;
                                                                                phi_4497_ = select(true, false, _e2949_);
                                                                            } else {
                                                                                phi_4496_ = f32();
                                                                                phi_4497_ = true;
                                                                            }
                                                                            let _e3037_ = phi_4496_;
                                                                            let _e3039_ = phi_4497_;
                                                                            let _e3040_ = select(_e3037_, 1000000000f, _e3039_);
                                                                            let _e3042_ = select(_e2941_, _e3040_, (_e3040_ > _e2941_));
                                                                            phi_4512_ = select(_e2839_, _e3042_, (_e3042_ > _e2839_));
                                                                        } else {
                                                                            phi_4512_ = f32();
                                                                        }
                                                                        let _e3046_ = phi_4512_;
                                                                        phi_4514_ = _e3046_;
                                                                        phi_4515_ = select(true, false, _e2842_);
                                                                    } else {
                                                                        phi_4514_ = f32();
                                                                        phi_4515_ = true;
                                                                    }
                                                                    let _e3049_ = phi_4514_;
                                                                    let _e3051_ = phi_4515_;
                                                                    phi_4520_ = select(_e3049_, _e2839_, _e3051_);
                                                                } else {
                                                                    phi_4520_ = 0f;
                                                                }
                                                                let _e3054_ = phi_4520_;
                                                                let _e3055_ = (_e163_ < 128u);
                                                                if _e3055_ {
                                                                    let _e3056_ = (32u + _e164_);
                                                                    if (_e3056_ < 256u) {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e3056_] = _e3054_;
                                                                }
                                                                workgroupBarrier();
                                                                if (_e163_ >= 128u) {
                                                                    let _e3060_ = (32u + _e164_);
                                                                    let _e3061_ = (_e3060_ < 256u);
                                                                    if _e3061_ {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    let _e3063_ = aux_scratch[_e3060_];
                                                                    if _e3061_ {
                                                                    } else {
                                                                        phi_34157_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e3060_] = select(_e3063_, _e3054_, (_e3054_ > _e3063_));
                                                                }
                                                                workgroupBarrier();
                                                                phi_4558_ = 64u;
                                                                loop {
                                                                    let _e3067_ = phi_4558_;
                                                                    let _e3068_ = (_e3067_ > 0u);
                                                                    if _e3068_ {
                                                                        if (_e164_ < _e3067_) {
                                                                            if _e3055_ {
                                                                                let _e3070_ = (32u + _e164_);
                                                                                let _e3071_ = (_e3070_ < 256u);
                                                                                if _e3071_ {
                                                                                } else {
                                                                                    phi_34066_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e3073_ = aux_scratch[_e3070_];
                                                                                let _e3074_ = (_e3070_ + _e3067_);
                                                                                if (_e3074_ < 256u) {
                                                                                } else {
                                                                                    phi_34066_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e3077_ = aux_scratch[_e3074_];
                                                                                if _e3071_ {
                                                                                } else {
                                                                                    phi_34066_ = true;
                                                                                    break;
                                                                                }
                                                                                aux_scratch[_e3070_] = select(_e3073_, _e3077_, (_e3077_ > _e3073_));
                                                                            }
                                                                        }
                                                                        workgroupBarrier();
                                                                        phi_4559_ = (_e3067_ >> bitcast<u32>(1i));
                                                                    } else {
                                                                        phi_4559_ = u32();
                                                                    }
                                                                    let _e3083_ = phi_4559_;
                                                                    phi_4558_ = _e3083_;
                                                                    phi_34066_ = _e2563_;
                                                                    if !(_e3068_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e3086_ = phi_34066_;
                                                                phi_34157_ = _e3086_;
                                                                if _e3086_ {
                                                                    break;
                                                                }
                                                                let _e3088_ = aux_scratch[32];
                                                                workgroupBarrier();
                                                                if (_e3088_ <= 0.0000005f) {
                                                                    if (_e2401_.member_1_ > 0u) {
                                                                        phi_4619_ = true;
                                                                    } else {
                                                                        phi_4619_ = (_e3088_ == 0f);
                                                                    }
                                                                    let _e3093_ = phi_4619_;
                                                                    phi_4624_ = select(_e2386_, true, _e3093_);
                                                                } else {
                                                                    phi_4624_ = _e2386_;
                                                                }
                                                                let _e3096_ = phi_4624_;
                                                                phi_34159_ = _e3086_;
                                                                phi_4625_ = _e2643_;
                                                                phi_4626_ = _e2601_[1];
                                                                phi_4627_ = _e2601_[0];
                                                                phi_4628_ = _e2643_;
                                                                phi_4629_ = _e3096_;
                                                            }
                                                            let _e3098_ = phi_34159_;
                                                            let _e3100_ = phi_4625_;
                                                            let _e3102_ = phi_4626_;
                                                            let _e3104_ = phi_4627_;
                                                            let _e3106_ = phi_4628_;
                                                            let _e3108_ = phi_4629_;
                                                            phi_34158_ = _e3098_;
                                                            phi_3498_ = _e3100_;
                                                            phi_3500_ = _e3102_;
                                                            phi_3502_ = _e3104_;
                                                            phi_3504_ = _e3106_;
                                                            phi_3506_ = _e3108_;
                                                        } else {
                                                            phi_34158_ = _e2374_;
                                                            phi_3498_ = array<f32, 2>();
                                                            phi_3500_ = f32();
                                                            phi_3502_ = f32();
                                                            phi_3504_ = array<f32, 2>();
                                                            phi_3506_ = bool();
                                                        }
                                                        let _e3110_ = phi_34158_;
                                                        let _e3112_ = phi_3498_;
                                                        let _e3114_ = phi_3500_;
                                                        let _e3116_ = phi_3502_;
                                                        let _e3118_ = phi_3504_;
                                                        let _e3120_ = phi_3506_;
                                                        phi_34048_ = _e3110_;
                                                        phi_3494_ = _e2399_;
                                                        phi_3497_ = _e3112_;
                                                        phi_3499_ = _e3114_;
                                                        phi_3501_ = _e3116_;
                                                        phi_3503_ = _e3118_;
                                                        phi_3505_ = _e3120_;
                                                        phi_34157_ = _e3110_;
                                                        if !(_e2405_) {
                                                            break;
                                                        }
                                                    }
                                                    let _e3123_ = phi_34157_;
                                                    phi_34247_ = _e3123_;
                                                    if _e3123_ {
                                                        break;
                                                    }
                                                    let _e3125_ = local_24_;
                                                    if _e3125_ {
                                                        if _e1583_ {
                                                            let _e3434_ = (_e2335_ * 0.5f);
                                                            if (_e3434_ != _e3434_) {
                                                                phi_24680_ = true;
                                                            } else {
                                                                phi_24680_ = (0.00001f >= _e3434_);
                                                            }
                                                            let _e3438_ = phi_24680_;
                                                            let _e3439_ = select(_e3434_, 0.00001f, _e3438_);
                                                            if (_e2293_ != _e2293_) {
                                                                phi_24695_ = true;
                                                            } else {
                                                                phi_24695_ = (0f >= _e2293_);
                                                            }
                                                            let _e3443_ = phi_24695_;
                                                            if (_e2342_ != _e2342_) {
                                                                phi_24710_ = true;
                                                            } else {
                                                                phi_24710_ = (0f >= _e2342_);
                                                            }
                                                            let _e3448_ = phi_24710_;
                                                            let _e3451_ = local_29_;
                                                            let _e3453_ = local_30_;
                                                            let _e3454_ = (_e3451_ * _e3453_);
                                                            if (_e3454_ != _e3454_) {
                                                                phi_24725_ = true;
                                                            } else {
                                                                phi_24725_ = (64f <= _e3454_);
                                                            }
                                                            let _e3458_ = phi_24725_;
                                                            let _e3461_ = (select(_e2293_, 0f, _e3443_) + (select(_e2342_, 0f, _e3448_) * select(_e3454_, 64f, _e3458_)));
                                                            let _e3464_ = (1f + (_e3439_ * (_e3439_ + _e3461_)));
                                                            let _e3467_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3464_)));
                                                            let _e3471_ = (_e3467_ + (_e3467_ * (1f - (_e3464_ * _e3467_))));
                                                            let _e3475_ = (_e3471_ + (_e3471_ * (1f - (_e3464_ * _e3471_))));
                                                            let _e3476_ = (_e3439_ * _e3475_);
                                                            let _e3480_ = local_31_;
                                                            let _e3481_ = (_e2361_ - _e3480_);
                                                            let _e3483_ = local_32_;
                                                            let _e3486_ = ((_e3475_ * _e3483_) + (_e3476_ * _e3481_));
                                                            let _e3488_ = local_33_;
                                                            let _e3491_ = local_34_;
                                                            let _e3494_ = ((_e3491_ + (_e3476_ * _e3488_)) + ((_e3439_ * _e3476_) * _e3481_));
                                                            let _e3509_ = select(0u, 2u, _e167_);
                                                            let _e3510_ = (_e2281_.member_1_ * 4u);
                                                            let _e3513_ = (_e164_ & 127u);
                                                            let _e3514_ = (((_e3510_ + _e3509_) * 128u) + _e3513_);
                                                            if (_e3514_ < 2048u) {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            lane_buf[_e3514_] = (((((_e3494_ * _e2300_) + (_e3486_ * _e2307_)) + (((_e2361_ - (_e3461_ * _e3486_)) - _e3494_) * _e2314_)) + ((_e2361_ - ((2f * _e3461_) * _e3486_)) * _e2321_)) + (_e2361_ * _e2328_));
                                                            let _e3520_ = (((_e3510_ + (_e3509_ + 1u)) * 128u) + _e3513_);
                                                            if (_e3520_ < 2048u) {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            lane_buf[_e3520_] = _e3486_;
                                                        }
                                                        if _e1184_ {
                                                            let _e3524_ = local_35_;
                                                            if (_e2281_.member_1_ < 256u) {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e2281_.member_1_] = _e3524_[0];
                                                            let _e3529_ = local_36_;
                                                            let _e3531_ = (4u + _e2281_.member_1_);
                                                            if (_e3531_ < 256u) {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e3531_] = _e3529_[1];
                                                        }
                                                        if (_e163_ == 128u) {
                                                            let _e3536_ = local_37_;
                                                            let _e3538_ = (8u + _e2281_.member_1_);
                                                            if (_e3538_ < 256u) {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e3538_] = _e3536_[0];
                                                            let _e3542_ = local_38_;
                                                            let _e3544_ = (12u + _e2281_.member_1_);
                                                            if (_e3544_ < 256u) {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e3544_] = _e3542_[1];
                                                        }
                                                        workgroupBarrier();
                                                        phi_34249_ = _e3123_;
                                                    } else {
                                                        let _e3126_ = select(0u, 128u, _e167_);
                                                        if _e1583_ {
                                                            let _e3131_ = (_e3126_ + _e164_);
                                                            if (_e3131_ < 1536u) {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            scan_scratch[_e3131_] = _e2361_;
                                                        } else {
                                                            if (_e164_ < 128u) {
                                                                let _e3128_ = (_e3126_ + _e164_);
                                                                if (_e3128_ < 1536u) {
                                                                } else {
                                                                    phi_34247_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e3128_] = 0f;
                                                            }
                                                        }
                                                        workgroupBarrier();
                                                        if _e1184_ {
                                                            let _e3134_ = (_e2281_.member_1_ < 256u);
                                                            if _e3134_ {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            let _e3136_ = aux_scratch[_e2281_.member_1_];
                                                            let _e3137_ = (4u + _e2281_.member_1_);
                                                            let _e3138_ = (_e3137_ < 256u);
                                                            if _e3138_ {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            let _e3140_ = aux_scratch[_e3137_];
                                                            phi_4992_ = type_15_(0u, _e1581_);
                                                            phi_4995_ = _e3140_;
                                                            phi_4997_ = _e3136_;
                                                            loop {
                                                                let _e3143_ = phi_4992_;
                                                                let _e3145_ = phi_4995_;
                                                                let _e3147_ = phi_4997_;
                                                                local_25_ = _e3147_;
                                                                local_26_ = _e3145_;
                                                                if (_e3143_.member < _e3143_.member_1_) {
                                                                    phi_4993_ = type_15_((_e3143_.member + 1u), _e3143_.member_1_);
                                                                    phi_5020_ = type_15_(1u, _e3143_.member);
                                                                } else {
                                                                    phi_4993_ = _e3143_;
                                                                    phi_5020_ = type_15_(0u, u32());
                                                                }
                                                                let _e3160_ = phi_4993_;
                                                                let _e3162_ = phi_5020_;
                                                                let _e3166_ = (bitcast<i32>(_e3162_.member) != 0i);
                                                                if _e3166_ {
                                                                    let _e3167_ = (_e3162_.member_1_ & 127u);
                                                                    if (_e3167_ < 1536u) {
                                                                    } else {
                                                                        phi_34163_ = true;
                                                                        break;
                                                                    }
                                                                    let _e3170_ = scan_scratch[_e3167_];
                                                                    let _e3171_ = (_e2335_ * 0.5f);
                                                                    if (_e3171_ != _e3171_) {
                                                                        phi_24200_ = true;
                                                                    } else {
                                                                        phi_24200_ = (0.00001f >= _e3171_);
                                                                    }
                                                                    let _e3175_ = phi_24200_;
                                                                    let _e3176_ = select(_e3171_, 0.00001f, _e3175_);
                                                                    if (_e2293_ != _e2293_) {
                                                                        phi_24260_ = true;
                                                                    } else {
                                                                        phi_24260_ = (0f >= _e2293_);
                                                                    }
                                                                    let _e3180_ = phi_24260_;
                                                                    if (_e2342_ != _e2342_) {
                                                                        phi_24320_ = true;
                                                                    } else {
                                                                        phi_24320_ = (0f >= _e2342_);
                                                                    }
                                                                    let _e3185_ = phi_24320_;
                                                                    let _e3187_ = (_e3147_ * _e3147_);
                                                                    if (_e3187_ != _e3187_) {
                                                                        phi_24380_ = true;
                                                                    } else {
                                                                        phi_24380_ = (64f <= _e3187_);
                                                                    }
                                                                    let _e3191_ = phi_24380_;
                                                                    let _e3194_ = (select(_e2293_, 0f, _e3180_) + (select(_e2342_, 0f, _e3185_) * select(_e3187_, 64f, _e3191_)));
                                                                    let _e3197_ = (1f + (_e3176_ * (_e3176_ + _e3194_)));
                                                                    let _e3200_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3197_)));
                                                                    let _e3204_ = (_e3200_ + (_e3200_ * (1f - (_e3197_ * _e3200_))));
                                                                    let _e3208_ = (_e3204_ + (_e3204_ * (1f - (_e3197_ * _e3204_))));
                                                                    let _e3209_ = (_e3176_ * _e3208_);
                                                                    let _e3212_ = (_e3170_ - _e3145_);
                                                                    let _e3215_ = ((_e3208_ * _e3147_) + (_e3209_ * _e3212_));
                                                                    let _e3219_ = ((_e3145_ + (_e3209_ * _e3147_)) + ((_e3176_ * _e3209_) * _e3212_));
                                                                    let _e3222_ = bitcast<u32>(((_e3215_ + _e3215_) - _e3147_));
                                                                    let _e3223_ = (_e3222_ & 2139095040u);
                                                                    if (_e3223_ == 2139095040u) {
                                                                        phi_5187_ = 0u;
                                                                    } else {
                                                                        if (_e3223_ == 0u) {
                                                                            phi_5186_ = (_e3222_ & 2147483648u);
                                                                        } else {
                                                                            phi_5186_ = _e3222_;
                                                                        }
                                                                        let _e3228_ = phi_5186_;
                                                                        phi_5187_ = _e3228_;
                                                                    }
                                                                    let _e3230_ = phi_5187_;
                                                                    let _e3234_ = bitcast<u32>(((_e3219_ + _e3219_) - _e3145_));
                                                                    let _e3235_ = (_e3234_ & 2139095040u);
                                                                    if (_e3235_ == 2139095040u) {
                                                                        phi_5239_ = 0u;
                                                                    } else {
                                                                        if (_e3235_ == 0u) {
                                                                            phi_5238_ = (_e3234_ & 2147483648u);
                                                                        } else {
                                                                            phi_5238_ = _e3234_;
                                                                        }
                                                                        let _e3240_ = phi_5238_;
                                                                        phi_5239_ = _e3240_;
                                                                    }
                                                                    let _e3242_ = phi_5239_;
                                                                    let _e3260_ = (_e3167_ & 127u);
                                                                    let _e3261_ = ((_e2281_.member_1_ * 512u) + _e3260_);
                                                                    if (_e3261_ < 2048u) {
                                                                    } else {
                                                                        phi_34163_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3261_] = (((((_e3219_ * _e2300_) + (_e3215_ * _e2307_)) + (((_e3170_ - (_e3194_ * _e3215_)) - _e3219_) * _e2314_)) + ((_e3170_ - ((_e3194_ + _e3194_) * _e3215_)) * _e2321_)) + (_e3170_ * _e2328_));
                                                                    let _e3266_ = ((((_e2281_.member_1_ * 4u) + 1u) * 128u) + _e3260_);
                                                                    if (_e3266_ < 2048u) {
                                                                    } else {
                                                                        phi_34163_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3266_] = _e3215_;
                                                                    phi_4996_ = bitcast<f32>(_e3242_);
                                                                    phi_4998_ = bitcast<f32>(_e3230_);
                                                                } else {
                                                                    phi_4996_ = f32();
                                                                    phi_4998_ = f32();
                                                                }
                                                                let _e3270_ = phi_4996_;
                                                                let _e3272_ = phi_4998_;
                                                                phi_4992_ = _e3160_;
                                                                phi_4995_ = _e3270_;
                                                                phi_4997_ = _e3272_;
                                                                phi_34163_ = _e3123_;
                                                                if !(_e3166_) {
                                                                    break;
                                                                }
                                                            }
                                                            let _e3275_ = phi_34163_;
                                                            phi_34247_ = _e3275_;
                                                            if _e3275_ {
                                                                break;
                                                            }
                                                            if _e3134_ {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            let _e3277_ = local_25_;
                                                            aux_scratch[_e2281_.member_1_] = _e3277_;
                                                            if _e3138_ {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            let _e3279_ = local_26_;
                                                            aux_scratch[_e3137_] = _e3279_;
                                                            phi_34210_ = _e3275_;
                                                        } else {
                                                            phi_34210_ = _e3123_;
                                                        }
                                                        let _e3281_ = phi_34210_;
                                                        if (_e163_ == 1u) {
                                                            let _e3283_ = (8u + _e2281_.member_1_);
                                                            let _e3284_ = (_e3283_ < 256u);
                                                            if _e3284_ {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            let _e3286_ = aux_scratch[_e3283_];
                                                            let _e3287_ = (12u + _e2281_.member_1_);
                                                            let _e3288_ = (_e3287_ < 256u);
                                                            if _e3288_ {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            let _e3290_ = aux_scratch[_e3287_];
                                                            phi_5371_ = type_15_(0u, _e1581_);
                                                            phi_5374_ = _e3290_;
                                                            phi_5376_ = _e3286_;
                                                            loop {
                                                                let _e3293_ = phi_5371_;
                                                                let _e3295_ = phi_5374_;
                                                                let _e3297_ = phi_5376_;
                                                                local_27_ = _e3297_;
                                                                local_28_ = _e3295_;
                                                                if (_e3293_.member < _e3293_.member_1_) {
                                                                    phi_5372_ = type_15_((_e3293_.member + 1u), _e3293_.member_1_);
                                                                    phi_5399_ = type_15_(1u, _e3293_.member);
                                                                } else {
                                                                    phi_5372_ = _e3293_;
                                                                    phi_5399_ = type_15_(0u, u32());
                                                                }
                                                                let _e3310_ = phi_5372_;
                                                                let _e3312_ = phi_5399_;
                                                                let _e3316_ = (bitcast<i32>(_e3312_.member) != 0i);
                                                                if _e3316_ {
                                                                    let _e3317_ = (_e3312_.member_1_ & 127u);
                                                                    let _e3318_ = (128u + _e3317_);
                                                                    if (_e3318_ < 1536u) {
                                                                    } else {
                                                                        phi_34206_ = true;
                                                                        break;
                                                                    }
                                                                    let _e3321_ = scan_scratch[_e3318_];
                                                                    let _e3322_ = (_e2335_ * 0.5f);
                                                                    if (_e3322_ != _e3322_) {
                                                                        phi_24440_ = true;
                                                                    } else {
                                                                        phi_24440_ = (0.00001f >= _e3322_);
                                                                    }
                                                                    let _e3326_ = phi_24440_;
                                                                    let _e3327_ = select(_e3322_, 0.00001f, _e3326_);
                                                                    if (_e2293_ != _e2293_) {
                                                                        phi_24500_ = true;
                                                                    } else {
                                                                        phi_24500_ = (0f >= _e2293_);
                                                                    }
                                                                    let _e3331_ = phi_24500_;
                                                                    if (_e2342_ != _e2342_) {
                                                                        phi_24560_ = true;
                                                                    } else {
                                                                        phi_24560_ = (0f >= _e2342_);
                                                                    }
                                                                    let _e3336_ = phi_24560_;
                                                                    let _e3338_ = (_e3297_ * _e3297_);
                                                                    if (_e3338_ != _e3338_) {
                                                                        phi_24620_ = true;
                                                                    } else {
                                                                        phi_24620_ = (64f <= _e3338_);
                                                                    }
                                                                    let _e3342_ = phi_24620_;
                                                                    let _e3345_ = (select(_e2293_, 0f, _e3331_) + (select(_e2342_, 0f, _e3336_) * select(_e3338_, 64f, _e3342_)));
                                                                    let _e3348_ = (1f + (_e3327_ * (_e3327_ + _e3345_)));
                                                                    let _e3351_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3348_)));
                                                                    let _e3355_ = (_e3351_ + (_e3351_ * (1f - (_e3348_ * _e3351_))));
                                                                    let _e3359_ = (_e3355_ + (_e3355_ * (1f - (_e3348_ * _e3355_))));
                                                                    let _e3360_ = (_e3327_ * _e3359_);
                                                                    let _e3363_ = (_e3321_ - _e3295_);
                                                                    let _e3366_ = ((_e3359_ * _e3297_) + (_e3360_ * _e3363_));
                                                                    let _e3370_ = ((_e3295_ + (_e3360_ * _e3297_)) + ((_e3327_ * _e3360_) * _e3363_));
                                                                    let _e3373_ = bitcast<u32>(((_e3366_ + _e3366_) - _e3297_));
                                                                    let _e3374_ = (_e3373_ & 2139095040u);
                                                                    if (_e3374_ == 2139095040u) {
                                                                        phi_5567_ = 0u;
                                                                    } else {
                                                                        if (_e3374_ == 0u) {
                                                                            phi_5566_ = (_e3373_ & 2147483648u);
                                                                        } else {
                                                                            phi_5566_ = _e3373_;
                                                                        }
                                                                        let _e3379_ = phi_5566_;
                                                                        phi_5567_ = _e3379_;
                                                                    }
                                                                    let _e3381_ = phi_5567_;
                                                                    let _e3385_ = bitcast<u32>(((_e3370_ + _e3370_) - _e3295_));
                                                                    let _e3386_ = (_e3385_ & 2139095040u);
                                                                    if (_e3386_ == 2139095040u) {
                                                                        phi_5619_ = 0u;
                                                                    } else {
                                                                        if (_e3386_ == 0u) {
                                                                            phi_5618_ = (_e3385_ & 2147483648u);
                                                                        } else {
                                                                            phi_5618_ = _e3385_;
                                                                        }
                                                                        let _e3391_ = phi_5618_;
                                                                        phi_5619_ = _e3391_;
                                                                    }
                                                                    let _e3393_ = phi_5619_;
                                                                    let _e3409_ = (_e2281_.member_1_ * 4u);
                                                                    let _e3412_ = (_e3317_ & 127u);
                                                                    let _e3413_ = (((_e3409_ + 2u) * 128u) + _e3412_);
                                                                    if (_e3413_ < 2048u) {
                                                                    } else {
                                                                        phi_34206_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3413_] = (((((_e3370_ * _e2300_) + (_e3366_ * _e2307_)) + (((_e3321_ - (_e3345_ * _e3366_)) - _e3370_) * _e2314_)) + ((_e3321_ - ((_e3345_ + _e3345_) * _e3366_)) * _e2321_)) + (_e3321_ * _e2328_));
                                                                    let _e3418_ = (((_e3409_ + 3u) * 128u) + _e3412_);
                                                                    if (_e3418_ < 2048u) {
                                                                    } else {
                                                                        phi_34206_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3418_] = _e3366_;
                                                                    phi_5375_ = bitcast<f32>(_e3393_);
                                                                    phi_5377_ = bitcast<f32>(_e3381_);
                                                                } else {
                                                                    phi_5375_ = f32();
                                                                    phi_5377_ = f32();
                                                                }
                                                                let _e3422_ = phi_5375_;
                                                                let _e3424_ = phi_5377_;
                                                                phi_5371_ = _e3310_;
                                                                phi_5374_ = _e3422_;
                                                                phi_5376_ = _e3424_;
                                                                phi_34206_ = _e3281_;
                                                                if !(_e3316_) {
                                                                    break;
                                                                }
                                                            }
                                                            let _e3427_ = phi_34206_;
                                                            phi_34247_ = _e3427_;
                                                            if _e3427_ {
                                                                break;
                                                            }
                                                            if _e3284_ {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            let _e3429_ = local_27_;
                                                            aux_scratch[_e3283_] = _e3429_;
                                                            if _e3288_ {
                                                            } else {
                                                                phi_34247_ = true;
                                                                break;
                                                            }
                                                            let _e3431_ = local_28_;
                                                            aux_scratch[_e3287_] = _e3431_;
                                                            phi_34250_ = _e3427_;
                                                        } else {
                                                            phi_34250_ = _e3281_;
                                                        }
                                                        let _e3433_ = phi_34250_;
                                                        workgroupBarrier();
                                                        phi_34249_ = _e3433_;
                                                    }
                                                    let _e3548_ = phi_34249_;
                                                    phi_34248_ = _e3548_;
                                                } else {
                                                    phi_34248_ = _e2264_;
                                                }
                                                let _e3550_ = phi_34248_;
                                                phi_34060_ = _e3550_;
                                                phi_2943_ = _e2279_;
                                                phi_34247_ = _e3550_;
                                                if !(_e2285_) {
                                                    break;
                                                }
                                            }
                                            let _e3553_ = phi_34247_;
                                            phi_36246_ = _e3553_;
                                            if _e3553_ {
                                                break;
                                            }
                                            phi_34291_ = _e3553_;
                                        } else {
                                            if (_e163_ < 128u) {
                                                let _e1954_ = vec4(_e1583_);
                                                let _e1955_ = select(vec4<f32>(0f, 0f, 0f, 0f), _e1604_, _e1954_);
                                                let _e1956_ = select(vec4<f32>(0f, 0f, 0f, 0f), _e1602_, _e1954_);
                                                phi_5744_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e1958_ = phi_5744_;
                                                    if (_e1958_.member < _e1958_.member_1_) {
                                                        phi_5745_ = type_15_((_e1958_.member + 1u), _e1958_.member_1_);
                                                        phi_5768_ = type_15_(1u, _e1958_.member);
                                                    } else {
                                                        phi_5745_ = _e1958_;
                                                        phi_5768_ = type_15_(0u, u32());
                                                    }
                                                    let _e1971_ = phi_5745_;
                                                    let _e1973_ = phi_5768_;
                                                    let _e1977_ = (bitcast<i32>(_e1973_.member) != 0i);
                                                    if _e1977_ {
                                                        let _e1978_ = (_e1973_.member_1_ & 3u);
                                                        switch bitcast<i32>(_e1978_) {
                                                            case 0: {
                                                                phi_5786_ = _e1955_.x;
                                                                break;
                                                            }
                                                            case 1: {
                                                                phi_5786_ = _e1955_.y;
                                                                break;
                                                            }
                                                            case 2: {
                                                                phi_5786_ = _e1955_.z;
                                                                break;
                                                            }
                                                            default: {
                                                                phi_5786_ = _e1955_.w;
                                                                break;
                                                            }
                                                        }
                                                        let _e1985_ = phi_5786_;
                                                        let _e1986_ = bitcast<u32>(_e1985_);
                                                        let _e1987_ = (_e1986_ & 2139095040u);
                                                        if (_e1987_ == 2139095040u) {
                                                            phi_5797_ = true;
                                                        } else {
                                                            phi_5797_ = (_e1987_ == 0u);
                                                        }
                                                        let _e1991_ = phi_5797_;
                                                        let _e1996_ = ((_e1973_.member_1_ * 256u) + _e164_);
                                                        if (_e1996_ < 1536u) {
                                                        } else {
                                                            phi_33970_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1996_] = bitcast<f32>(select(_e1986_, 0u, _e1991_));
                                                        switch bitcast<i32>(_e1978_) {
                                                            case 0: {
                                                                phi_5822_ = _e1956_.x;
                                                                break;
                                                            }
                                                            case 1: {
                                                                phi_5822_ = _e1956_.y;
                                                                break;
                                                            }
                                                            case 2: {
                                                                phi_5822_ = _e1956_.z;
                                                                break;
                                                            }
                                                            default: {
                                                                phi_5822_ = _e1956_.w;
                                                                break;
                                                            }
                                                        }
                                                        let _e2005_ = phi_5822_;
                                                        let _e2006_ = bitcast<u32>(_e2005_);
                                                        let _e2007_ = (_e2006_ & 2139095040u);
                                                        if (_e2007_ == 2139095040u) {
                                                            phi_5833_ = true;
                                                        } else {
                                                            phi_5833_ = (_e2007_ == 0u);
                                                        }
                                                        let _e2011_ = phi_5833_;
                                                        let _e2016_ = ((((_e1973_.member_1_ * 2u) + 1u) * 128u) + _e164_);
                                                        if (_e2016_ < 1536u) {
                                                        } else {
                                                            phi_33970_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e2016_] = bitcast<f32>(select(_e2006_, 0u, _e2011_));
                                                    }
                                                    phi_5744_ = _e1971_;
                                                    phi_33970_ = _e1552_;
                                                    if !(_e1977_) {
                                                        break;
                                                    }
                                                }
                                                let _e2021_ = phi_33970_;
                                                phi_36246_ = _e2021_;
                                                if _e2021_ {
                                                    break;
                                                }
                                                phi_33997_ = _e2021_;
                                            } else {
                                                phi_33997_ = _e1552_;
                                            }
                                            let _e2023_ = phi_33997_;
                                            workgroupBarrier();
                                            if (_e163_ < 8u) {
                                                let _e2026_ = (_e163_ >> bitcast<u32>(1i));
                                                let _e2027_ = (_e163_ & 1u);
                                                let _e2028_ = (_e2027_ != 0u);
                                                let _e2029_ = (_e2026_ & 3u);
                                                switch bitcast<i32>(_e2029_) {
                                                    case 0: {
                                                        phi_5961_ = _e327_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5961_ = _e327_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5961_ = _e327_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5961_ = _e327_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2036_ = phi_5961_;
                                                switch bitcast<i32>(_e2029_) {
                                                    case 0: {
                                                        phi_5978_ = _e341_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5978_ = _e341_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5978_ = _e341_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5978_ = _e341_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2043_ = phi_5978_;
                                                switch bitcast<i32>(_e2029_) {
                                                    case 0: {
                                                        phi_5989_ = _e348_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5989_ = _e348_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5989_ = _e348_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5989_ = _e348_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2050_ = phi_5989_;
                                                switch bitcast<i32>(_e2029_) {
                                                    case 0: {
                                                        phi_6000_ = _e355_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6000_ = _e355_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6000_ = _e355_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6000_ = _e355_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2057_ = phi_6000_;
                                                switch bitcast<i32>(_e2029_) {
                                                    case 0: {
                                                        phi_6011_ = _e362_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6011_ = _e362_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6011_ = _e362_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6011_ = _e362_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2064_ = phi_6011_;
                                                switch bitcast<i32>(_e2029_) {
                                                    case 0: {
                                                        phi_6022_ = _e369_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6022_ = _e369_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6022_ = _e369_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6022_ = _e369_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2071_ = phi_6022_;
                                                switch bitcast<i32>(_e2029_) {
                                                    case 0: {
                                                        phi_6237_ = _e418_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6237_ = _e418_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6237_ = _e418_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6237_ = _e418_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2078_ = phi_6237_;
                                                switch bitcast<i32>(_e2029_) {
                                                    case 0: {
                                                        phi_6312_ = _e469_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6312_ = _e469_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6312_ = _e469_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6312_ = _e469_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2085_ = phi_6312_;
                                                if _e2028_ {
                                                    phi_6317_ = (8u + _e2026_);
                                                } else {
                                                    phi_6317_ = _e2026_;
                                                }
                                                let _e2088_ = phi_6317_;
                                                if _e2028_ {
                                                    phi_6323_ = (12u + _e2026_);
                                                } else {
                                                    phi_6323_ = (4u + _e2026_);
                                                }
                                                let _e2092_ = phi_6323_;
                                                let _e2093_ = (_e2088_ < 256u);
                                                if _e2093_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                let _e2095_ = aux_scratch[_e2088_];
                                                let _e2096_ = (_e2092_ < 256u);
                                                if _e2096_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                let _e2098_ = aux_scratch[_e2092_];
                                                let _e2102_ = select(0u, 2u, _e2028_);
                                                phi_6348_ = type_15_(0u, _e1581_);
                                                phi_6351_ = _e2098_;
                                                phi_6353_ = _e2095_;
                                                loop {
                                                    let _e2105_ = phi_6348_;
                                                    let _e2107_ = phi_6351_;
                                                    let _e2109_ = phi_6353_;
                                                    local_22_ = _e2109_;
                                                    local_23_ = _e2107_;
                                                    if (_e2105_.member < _e2105_.member_1_) {
                                                        phi_6349_ = type_15_((_e2105_.member + 1u), _e2105_.member_1_);
                                                        phi_6376_ = type_15_(1u, _e2105_.member);
                                                    } else {
                                                        phi_6349_ = _e2105_;
                                                        phi_6376_ = type_15_(0u, u32());
                                                    }
                                                    let _e2122_ = phi_6349_;
                                                    let _e2124_ = phi_6376_;
                                                    let _e2128_ = (bitcast<i32>(_e2124_.member) != 0i);
                                                    if _e2128_ {
                                                        let _e2129_ = (_e2124_.member_1_ & 127u);
                                                        let _e2130_ = ((((_e2026_ * 2u) + _e2027_) * 128u) + _e2129_);
                                                        if (_e2130_ < 1536u) {
                                                        } else {
                                                            phi_33982_ = true;
                                                            break;
                                                        }
                                                        let _e2133_ = scan_scratch[_e2130_];
                                                        let _e2134_ = (_e2078_ * 0.5f);
                                                        if (_e2134_ != _e2134_) {
                                                            phi_23420_ = true;
                                                        } else {
                                                            phi_23420_ = (0.00001f >= _e2134_);
                                                        }
                                                        let _e2138_ = phi_23420_;
                                                        let _e2139_ = select(_e2134_, 0.00001f, _e2138_);
                                                        if (_e2036_ != _e2036_) {
                                                            phi_23480_ = true;
                                                        } else {
                                                            phi_23480_ = (0f >= _e2036_);
                                                        }
                                                        let _e2143_ = phi_23480_;
                                                        if (_e2085_ != _e2085_) {
                                                            phi_23540_ = true;
                                                        } else {
                                                            phi_23540_ = (0f >= _e2085_);
                                                        }
                                                        let _e2148_ = phi_23540_;
                                                        let _e2150_ = (_e2109_ * _e2109_);
                                                        if (_e2150_ != _e2150_) {
                                                            phi_23600_ = true;
                                                        } else {
                                                            phi_23600_ = (64f <= _e2150_);
                                                        }
                                                        let _e2154_ = phi_23600_;
                                                        let _e2157_ = (select(_e2036_, 0f, _e2143_) + (select(_e2085_, 0f, _e2148_) * select(_e2150_, 64f, _e2154_)));
                                                        let _e2160_ = (1f + (_e2139_ * (_e2139_ + _e2157_)));
                                                        let _e2163_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2160_)));
                                                        let _e2167_ = (_e2163_ + (_e2163_ * (1f - (_e2160_ * _e2163_))));
                                                        let _e2171_ = (_e2167_ + (_e2167_ * (1f - (_e2160_ * _e2167_))));
                                                        let _e2172_ = (_e2139_ * _e2171_);
                                                        let _e2175_ = (_e2133_ - _e2107_);
                                                        let _e2178_ = ((_e2171_ * _e2109_) + (_e2172_ * _e2175_));
                                                        let _e2182_ = ((_e2107_ + (_e2172_ * _e2109_)) + ((_e2139_ * _e2172_) * _e2175_));
                                                        let _e2185_ = bitcast<u32>(((_e2178_ + _e2178_) - _e2109_));
                                                        let _e2186_ = (_e2185_ & 2139095040u);
                                                        if (_e2186_ == 2139095040u) {
                                                            phi_6544_ = 0u;
                                                        } else {
                                                            if (_e2186_ == 0u) {
                                                                phi_6543_ = (_e2185_ & 2147483648u);
                                                            } else {
                                                                phi_6543_ = _e2185_;
                                                            }
                                                            let _e2191_ = phi_6543_;
                                                            phi_6544_ = _e2191_;
                                                        }
                                                        let _e2193_ = phi_6544_;
                                                        let _e2197_ = bitcast<u32>(((_e2182_ + _e2182_) - _e2107_));
                                                        let _e2198_ = (_e2197_ & 2139095040u);
                                                        if (_e2198_ == 2139095040u) {
                                                            phi_6596_ = 0u;
                                                        } else {
                                                            if (_e2198_ == 0u) {
                                                                phi_6595_ = (_e2197_ & 2147483648u);
                                                            } else {
                                                                phi_6595_ = _e2197_;
                                                            }
                                                            let _e2203_ = phi_6595_;
                                                            phi_6596_ = _e2203_;
                                                        }
                                                        let _e2205_ = phi_6596_;
                                                        let _e2221_ = (_e2026_ * 4u);
                                                        let _e2224_ = (_e2129_ & 127u);
                                                        let _e2225_ = (((_e2221_ + _e2102_) * 128u) + _e2224_);
                                                        if (_e2225_ < 2048u) {
                                                        } else {
                                                            phi_33982_ = true;
                                                            break;
                                                        }
                                                        lane_buf[_e2225_] = (((((_e2182_ * _e2043_) + (_e2178_ * _e2050_)) + (((_e2133_ - (_e2157_ * _e2178_)) - _e2182_) * _e2057_)) + ((_e2133_ - ((_e2157_ + _e2157_) * _e2178_)) * _e2064_)) + (_e2133_ * _e2071_));
                                                        let _e2231_ = (((_e2221_ + (_e2102_ + 1u)) * 128u) + _e2224_);
                                                        if (_e2231_ < 2048u) {
                                                        } else {
                                                            phi_33982_ = true;
                                                            break;
                                                        }
                                                        lane_buf[_e2231_] = _e2178_;
                                                        phi_6352_ = bitcast<f32>(_e2205_);
                                                        phi_6354_ = bitcast<f32>(_e2193_);
                                                    } else {
                                                        phi_6352_ = f32();
                                                        phi_6354_ = f32();
                                                    }
                                                    let _e2235_ = phi_6352_;
                                                    let _e2237_ = phi_6354_;
                                                    phi_6348_ = _e2122_;
                                                    phi_6351_ = _e2235_;
                                                    phi_6353_ = _e2237_;
                                                    phi_33982_ = _e2023_;
                                                    if !(_e2128_) {
                                                        break;
                                                    }
                                                }
                                                let _e2240_ = phi_33982_;
                                                phi_36246_ = _e2240_;
                                                if _e2240_ {
                                                    break;
                                                }
                                                let _e2242_ = local_22_;
                                                let _e2243_ = bitcast<u32>(_e2242_);
                                                let _e2244_ = (_e2243_ & 2139095040u);
                                                if (_e2244_ == 2139095040u) {
                                                    phi_6704_ = true;
                                                } else {
                                                    phi_6704_ = (_e2244_ == 0u);
                                                }
                                                let _e2248_ = phi_6704_;
                                                if _e2093_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                aux_scratch[_e2088_] = bitcast<f32>(select(_e2243_, 0u, _e2248_));
                                                let _e2252_ = local_23_;
                                                let _e2253_ = bitcast<u32>(_e2252_);
                                                let _e2254_ = (_e2253_ & 2139095040u);
                                                if (_e2254_ == 2139095040u) {
                                                    phi_6726_ = true;
                                                } else {
                                                    phi_6726_ = (_e2254_ == 0u);
                                                }
                                                let _e2258_ = phi_6726_;
                                                if _e2096_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                aux_scratch[_e2092_] = bitcast<f32>(select(_e2253_, 0u, _e2258_));
                                                phi_34292_ = _e2240_;
                                            } else {
                                                phi_34292_ = _e2023_;
                                            }
                                            let _e2262_ = phi_34292_;
                                            workgroupBarrier();
                                            phi_34291_ = _e2262_;
                                        }
                                        let _e3555_ = phi_34291_;
                                        phi_34290_ = _e3555_;
                                    } else {
                                        phi_33858_ = _e1552_;
                                        phi_6740_ = type_15_(0u, 4u);
                                        loop {
                                            let _e1606_ = phi_33858_;
                                            let _e1608_ = phi_6740_;
                                            if (_e1608_.member < _e1608_.member_1_) {
                                                phi_6741_ = type_15_((_e1608_.member + 1u), _e1608_.member_1_);
                                                phi_6764_ = type_15_(1u, _e1608_.member);
                                            } else {
                                                phi_6741_ = _e1608_;
                                                phi_6764_ = type_15_(0u, u32());
                                            }
                                            let _e1621_ = phi_6741_;
                                            let _e1623_ = phi_6764_;
                                            let _e1627_ = (bitcast<i32>(_e1623_.member) != 0i);
                                            if _e1627_ {
                                                let _e1628_ = (_e1623_.member_1_ & 3u);
                                                switch bitcast<i32>(_e1628_) {
                                                    case 0: {
                                                        phi_6859_ = _e306_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6859_ = _e306_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6859_ = _e306_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6859_ = _e306_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1635_ = phi_6859_;
                                                switch bitcast<i32>(_e1628_) {
                                                    case 0: {
                                                        phi_6870_ = _e313_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6870_ = _e313_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6870_ = _e313_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6870_ = _e313_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1642_ = phi_6870_;
                                                switch bitcast<i32>(_e1628_) {
                                                    case 0: {
                                                        phi_6881_ = _e320_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6881_ = _e320_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6881_ = _e320_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6881_ = _e320_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1649_ = phi_6881_;
                                                switch bitcast<i32>(_e1628_) {
                                                    case 0: {
                                                        phi_6892_ = _e327_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6892_ = _e327_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6892_ = _e327_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6892_ = _e327_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1656_ = phi_6892_;
                                                switch bitcast<i32>(_e1628_) {
                                                    case 0: {
                                                        phi_6903_ = _e334_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6903_ = _e334_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6903_ = _e334_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6903_ = _e334_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1663_ = phi_6903_;
                                                switch bitcast<i32>(_e1628_) {
                                                    case 0: {
                                                        phi_6914_ = _e341_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6914_ = _e341_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6914_ = _e341_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6914_ = _e341_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1670_ = phi_6914_;
                                                switch bitcast<i32>(_e1628_) {
                                                    case 0: {
                                                        phi_6925_ = _e348_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6925_ = _e348_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6925_ = _e348_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6925_ = _e348_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1677_ = phi_6925_;
                                                switch bitcast<i32>(_e1628_) {
                                                    case 0: {
                                                        phi_6936_ = _e355_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6936_ = _e355_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6936_ = _e355_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6936_ = _e355_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1684_ = phi_6936_;
                                                switch bitcast<i32>(_e1628_) {
                                                    case 0: {
                                                        phi_6947_ = _e362_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6947_ = _e362_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6947_ = _e362_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6947_ = _e362_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1691_ = phi_6947_;
                                                switch bitcast<i32>(_e1628_) {
                                                    case 0: {
                                                        phi_6958_ = _e369_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6958_ = _e369_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6958_ = _e369_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6958_ = _e369_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1698_ = phi_6958_;
                                                if _e1583_ {
                                                    let _e1700_ = select(_e1604_, _e1602_, vec4(_e167_));
                                                    switch bitcast<i32>(_e1628_) {
                                                        case 0: {
                                                            phi_7266_ = _e1700_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_7266_ = _e1700_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_7266_ = _e1700_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_7266_ = _e1700_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e1707_ = phi_7266_;
                                                    let _e1708_ = bitcast<u32>(_e1707_);
                                                    let _e1709_ = (_e1708_ & 2139095040u);
                                                    if (_e1709_ == 2139095040u) {
                                                        phi_7277_ = true;
                                                    } else {
                                                        phi_7277_ = (_e1709_ == 0u);
                                                    }
                                                    let _e1713_ = phi_7277_;
                                                    phi_7283_ = bitcast<f32>(select(_e1708_, 0u, _e1713_));
                                                } else {
                                                    phi_7283_ = 0f;
                                                }
                                                let _e1717_ = phi_7283_;
                                                if _e167_ {
                                                    phi_7288_ = (8u + _e1623_.member_1_);
                                                } else {
                                                    phi_7288_ = _e1623_.member_1_;
                                                }
                                                let _e1720_ = phi_7288_;
                                                if _e167_ {
                                                    phi_7294_ = (12u + _e1623_.member_1_);
                                                } else {
                                                    phi_7294_ = (4u + _e1623_.member_1_);
                                                }
                                                let _e1724_ = phi_7294_;
                                                if (_e1720_ < 256u) {
                                                } else {
                                                    phi_33953_ = true;
                                                    break;
                                                }
                                                let _e1727_ = aux_scratch[_e1720_];
                                                if (_e1724_ < 256u) {
                                                } else {
                                                    phi_33953_ = true;
                                                    break;
                                                }
                                                let _e1730_ = aux_scratch[_e1724_];
                                                let _e1732_ = select(0u, 256u, _e167_);
                                                let _e1733_ = (_e1642_ + _e1642_);
                                                let _e1734_ = (_e1649_ + _e1649_);
                                                let _e1736_ = ((_e1635_ + _e1635_) - 1f);
                                                let _e1737_ = -(_e1733_);
                                                let _e1738_ = (1f - _e1734_);
                                                if _e1583_ {
                                                    phi_7327_ = (_e1733_ * _e1717_);
                                                } else {
                                                    phi_7327_ = 0f;
                                                }
                                                let _e1741_ = phi_7327_;
                                                if _e1583_ {
                                                    phi_7332_ = (_e1734_ * _e1717_);
                                                } else {
                                                    phi_7332_ = 0f;
                                                }
                                                let _e1744_ = phi_7332_;
                                                let _e1745_ = (_e164_ == 0u);
                                                if _e1745_ {
                                                    if _e1582_ {
                                                        phi_7349_ = (_e1744_ + ((_e1733_ * _e1727_) + (_e1738_ * _e1730_)));
                                                        phi_7350_ = (_e1741_ + ((_e1736_ * _e1727_) + (_e1737_ * _e1730_)));
                                                    } else {
                                                        phi_7349_ = _e1744_;
                                                        phi_7350_ = _e1741_;
                                                    }
                                                    let _e1755_ = phi_7349_;
                                                    let _e1757_ = phi_7350_;
                                                    phi_7351_ = _e1755_;
                                                    phi_7352_ = _e1757_;
                                                } else {
                                                    phi_7351_ = _e1744_;
                                                    phi_7352_ = _e1741_;
                                                }
                                                let _e1759_ = phi_7351_;
                                                let _e1761_ = phi_7352_;
                                                let _e1762_ = (_e1732_ + _e164_);
                                                let _e1763_ = (_e1762_ < 1536u);
                                                if _e1763_ {
                                                } else {
                                                    phi_33953_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e1762_] = _e1761_;
                                                let _e1765_ = (_e1732_ + 128u);
                                                let _e1766_ = (_e1765_ + _e164_);
                                                let _e1767_ = (_e1766_ < 1536u);
                                                if _e1767_ {
                                                } else {
                                                    phi_33953_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e1766_] = _e1759_;
                                                workgroupBarrier();
                                                phi_7369_ = _e1738_;
                                                phi_7372_ = _e1733_;
                                                phi_7374_ = _e1737_;
                                                phi_7376_ = _e1736_;
                                                phi_7378_ = 1u;
                                                phi_7380_ = _e1759_;
                                                phi_7382_ = _e1761_;
                                                loop {
                                                    let _e1770_ = phi_7369_;
                                                    let _e1772_ = phi_7372_;
                                                    let _e1774_ = phi_7374_;
                                                    let _e1776_ = phi_7376_;
                                                    let _e1778_ = phi_7378_;
                                                    let _e1780_ = phi_7380_;
                                                    let _e1782_ = phi_7382_;
                                                    let _e1783_ = (_e1778_ < 128u);
                                                    if _e1783_ {
                                                        if (_e164_ >= _e1778_) {
                                                            let _e1786_ = ((_e164_ - _e1778_) & 127u);
                                                            let _e1787_ = (_e1732_ + _e1786_);
                                                            if (_e1787_ < 1536u) {
                                                            } else {
                                                                phi_33837_ = true;
                                                                break;
                                                            }
                                                            let _e1790_ = scan_scratch[_e1787_];
                                                            let _e1791_ = (_e1765_ + _e1786_);
                                                            if (_e1791_ < 1536u) {
                                                            } else {
                                                                phi_33837_ = true;
                                                                break;
                                                            }
                                                            let _e1794_ = scan_scratch[_e1791_];
                                                            phi_7421_ = (((_e1772_ * _e1790_) + (_e1770_ * _e1794_)) + _e1780_);
                                                            phi_7422_ = (((_e1776_ * _e1790_) + (_e1774_ * _e1794_)) + _e1782_);
                                                        } else {
                                                            phi_7421_ = _e1780_;
                                                            phi_7422_ = _e1782_;
                                                        }
                                                        let _e1804_ = phi_7421_;
                                                        let _e1806_ = phi_7422_;
                                                        workgroupBarrier();
                                                        if _e1763_ {
                                                        } else {
                                                            phi_33837_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1762_] = _e1806_;
                                                        if _e1767_ {
                                                        } else {
                                                            phi_33837_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1766_] = _e1804_;
                                                        let _e1810_ = (_e1776_ + _e1770_);
                                                        workgroupBarrier();
                                                        phi_7370_ = ((_e1772_ * _e1774_) + (_e1770_ * _e1770_));
                                                        phi_7373_ = (_e1772_ * _e1810_);
                                                        phi_7375_ = (_e1774_ * _e1810_);
                                                        phi_7377_ = ((_e1776_ * _e1776_) + (_e1774_ * _e1772_));
                                                        phi_7379_ = (_e1778_ << bitcast<u32>(1i));
                                                        phi_7381_ = _e1804_;
                                                        phi_7383_ = _e1806_;
                                                    } else {
                                                        phi_7370_ = f32();
                                                        phi_7373_ = f32();
                                                        phi_7375_ = f32();
                                                        phi_7377_ = f32();
                                                        phi_7379_ = u32();
                                                        phi_7381_ = f32();
                                                        phi_7383_ = f32();
                                                    }
                                                    let _e1819_ = phi_7370_;
                                                    let _e1821_ = phi_7373_;
                                                    let _e1823_ = phi_7375_;
                                                    let _e1825_ = phi_7377_;
                                                    let _e1827_ = phi_7379_;
                                                    let _e1829_ = phi_7381_;
                                                    let _e1831_ = phi_7383_;
                                                    phi_7369_ = _e1819_;
                                                    phi_7372_ = _e1821_;
                                                    phi_7374_ = _e1823_;
                                                    phi_7376_ = _e1825_;
                                                    phi_7378_ = _e1827_;
                                                    phi_7380_ = _e1829_;
                                                    phi_7382_ = _e1831_;
                                                    phi_33837_ = _e1606_;
                                                    if !(_e1783_) {
                                                        break;
                                                    }
                                                }
                                                let _e1834_ = phi_33837_;
                                                phi_33953_ = _e1834_;
                                                if _e1834_ {
                                                    break;
                                                }
                                                if _e1745_ {
                                                    phi_7474_ = _e1730_;
                                                    phi_7475_ = _e1727_;
                                                } else {
                                                    let _e1836_ = ((_e164_ - 1u) & 127u);
                                                    let _e1837_ = (_e1732_ + _e1836_);
                                                    if (_e1837_ < 1536u) {
                                                    } else {
                                                        phi_33953_ = true;
                                                        break;
                                                    }
                                                    let _e1840_ = scan_scratch[_e1837_];
                                                    let _e1841_ = (_e1765_ + _e1836_);
                                                    if (_e1841_ < 1536u) {
                                                    } else {
                                                        phi_33953_ = true;
                                                        break;
                                                    }
                                                    let _e1844_ = scan_scratch[_e1841_];
                                                    phi_7474_ = _e1844_;
                                                    phi_7475_ = _e1840_;
                                                }
                                                let _e1846_ = phi_7474_;
                                                let _e1848_ = phi_7475_;
                                                if _e1582_ {
                                                    if (_e1581_ <= 128u) {
                                                        phi_7485_ = (_e1581_ - 1u);
                                                    } else {
                                                        phi_7485_ = 127u;
                                                    }
                                                    let _e1852_ = phi_7485_;
                                                    let _e1853_ = (_e1732_ + _e1852_);
                                                    if (_e1853_ < 1536u) {
                                                    } else {
                                                        phi_33953_ = true;
                                                        break;
                                                    }
                                                    let _e1856_ = scan_scratch[_e1853_];
                                                    let _e1857_ = (_e1765_ + _e1852_);
                                                    if (_e1857_ < 1536u) {
                                                    } else {
                                                        phi_33953_ = true;
                                                        break;
                                                    }
                                                    let _e1860_ = scan_scratch[_e1857_];
                                                    phi_7504_ = array<f32, 2>(_e1856_, _e1860_);
                                                } else {
                                                    phi_7504_ = array<f32, 2>(_e1727_, _e1730_);
                                                }
                                                let _e1863_ = phi_7504_;
                                                workgroupBarrier();
                                                if _e1583_ {
                                                    let _e1864_ = (_e1717_ - _e1846_);
                                                    let _e1867_ = ((_e1635_ * _e1848_) + (_e1642_ * _e1864_));
                                                    let _e1871_ = ((_e1846_ + (_e1642_ * _e1848_)) + (_e1649_ * _e1864_));
                                                    let _e1886_ = select(0u, 2u, _e167_);
                                                    let _e1887_ = (_e1623_.member_1_ * 4u);
                                                    let _e1890_ = (_e164_ & 127u);
                                                    let _e1891_ = (((_e1887_ + _e1886_) * 128u) + _e1890_);
                                                    if (_e1891_ < 2048u) {
                                                    } else {
                                                        phi_33953_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e1891_] = (((((_e1871_ * _e1670_) + (_e1867_ * _e1677_)) + (((_e1717_ - (_e1656_ * _e1867_)) - _e1871_) * _e1684_)) + ((_e1717_ - (_e1663_ * _e1867_)) * _e1691_)) + (_e1717_ * _e1698_));
                                                    let _e1897_ = (((_e1887_ + (_e1886_ + 1u)) * 128u) + _e1890_);
                                                    if (_e1897_ < 2048u) {
                                                    } else {
                                                        phi_33953_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e1897_] = _e1867_;
                                                }
                                                if _e1184_ {
                                                    let _e1901_ = bitcast<u32>(_e1863_[0]);
                                                    let _e1902_ = (_e1901_ & 2139095040u);
                                                    if (_e1902_ == 2139095040u) {
                                                        phi_7759_ = true;
                                                    } else {
                                                        phi_7759_ = (_e1902_ == 0u);
                                                    }
                                                    let _e1906_ = phi_7759_;
                                                    if (_e1623_.member_1_ < 256u) {
                                                    } else {
                                                        phi_33953_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1623_.member_1_] = bitcast<f32>(select(_e1901_, 0u, _e1906_));
                                                    let _e1912_ = bitcast<u32>(_e1863_[1]);
                                                    let _e1913_ = (_e1912_ & 2139095040u);
                                                    if (_e1913_ == 2139095040u) {
                                                        phi_7782_ = true;
                                                    } else {
                                                        phi_7782_ = (_e1913_ == 0u);
                                                    }
                                                    let _e1917_ = phi_7782_;
                                                    let _e1920_ = (4u + _e1623_.member_1_);
                                                    if (_e1920_ < 256u) {
                                                    } else {
                                                        phi_33953_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1920_] = bitcast<f32>(select(_e1912_, 0u, _e1917_));
                                                }
                                                if (_e163_ == 128u) {
                                                    let _e1925_ = bitcast<u32>(_e1863_[0]);
                                                    let _e1926_ = (_e1925_ & 2139095040u);
                                                    if (_e1926_ == 2139095040u) {
                                                        phi_7810_ = true;
                                                    } else {
                                                        phi_7810_ = (_e1926_ == 0u);
                                                    }
                                                    let _e1930_ = phi_7810_;
                                                    let _e1933_ = (8u + _e1623_.member_1_);
                                                    if (_e1933_ < 256u) {
                                                    } else {
                                                        phi_33953_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1933_] = bitcast<f32>(select(_e1925_, 0u, _e1930_));
                                                    let _e1937_ = bitcast<u32>(_e1863_[1]);
                                                    let _e1938_ = (_e1937_ & 2139095040u);
                                                    if (_e1938_ == 2139095040u) {
                                                        phi_7834_ = true;
                                                    } else {
                                                        phi_7834_ = (_e1938_ == 0u);
                                                    }
                                                    let _e1942_ = phi_7834_;
                                                    let _e1945_ = (12u + _e1623_.member_1_);
                                                    if (_e1945_ < 256u) {
                                                    } else {
                                                        phi_33953_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1945_] = bitcast<f32>(select(_e1937_, 0u, _e1942_));
                                                }
                                                workgroupBarrier();
                                                phi_33954_ = _e1834_;
                                            } else {
                                                phi_33954_ = _e1606_;
                                            }
                                            let _e1949_ = phi_33954_;
                                            phi_33858_ = _e1949_;
                                            phi_6740_ = _e1621_;
                                            phi_33953_ = _e1949_;
                                            if !(_e1627_) {
                                                break;
                                            }
                                        }
                                        let _e1952_ = phi_33953_;
                                        phi_36246_ = _e1952_;
                                        if _e1952_ {
                                            break;
                                        }
                                        phi_34290_ = _e1952_;
                                    }
                                    let _e3557_ = phi_34290_;
                                    phi_34289_ = _e3557_;
                                    phi_7849_ = type_15_(0u, 4u);
                                    loop {
                                        let _e3559_ = phi_34289_;
                                        let _e3561_ = phi_7849_;
                                        if (_e3561_.member < _e3561_.member_1_) {
                                            phi_7850_ = type_15_((_e3561_.member + 1u), _e3561_.member_1_);
                                            phi_7873_ = type_15_(1u, _e3561_.member);
                                        } else {
                                            phi_7850_ = _e3561_;
                                            phi_7873_ = type_15_(0u, u32());
                                        }
                                        let _e3574_ = phi_7850_;
                                        let _e3576_ = phi_7873_;
                                        let _e3580_ = (bitcast<i32>(_e3576_.member) != 0i);
                                        if _e3580_ {
                                            let _e3581_ = (_e3576_.member_1_ & 3u);
                                            switch bitcast<i32>(_e3581_) {
                                                case 0: {
                                                    phi_7891_ = _e376_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_7891_ = _e376_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_7891_ = _e376_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_7891_ = _e376_.w;
                                                    break;
                                                }
                                            }
                                            let _e3588_ = phi_7891_;
                                            switch bitcast<i32>(_e3581_) {
                                                case 0: {
                                                    phi_7902_ = _e383_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_7902_ = _e383_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_7902_ = _e383_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_7902_ = _e383_.w;
                                                    break;
                                                }
                                            }
                                            let _e3595_ = phi_7902_;
                                            switch bitcast<i32>(_e3581_) {
                                                case 0: {
                                                    phi_7913_ = _e390_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_7913_ = _e390_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_7913_ = _e390_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_7913_ = _e390_.w;
                                                    break;
                                                }
                                            }
                                            let _e3602_ = phi_7913_;
                                            switch bitcast<i32>(_e3581_) {
                                                case 0: {
                                                    phi_7924_ = _e397_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_7924_ = _e397_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_7924_ = _e397_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_7924_ = _e397_.w;
                                                    break;
                                                }
                                            }
                                            let _e3609_ = phi_7924_;
                                            switch bitcast<i32>(_e3581_) {
                                                case 0: {
                                                    phi_7935_ = _e404_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_7935_ = _e404_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_7935_ = _e404_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_7935_ = _e404_.w;
                                                    break;
                                                }
                                            }
                                            let _e3616_ = phi_7935_;
                                            let _e3617_ = bitcast<u32>(_e3588_);
                                            let _e3618_ = (_e3617_ & 2139095040u);
                                            if (_e3618_ != 2139095040u) {
                                                if (_e3588_ <= 0f) {
                                                    phi_8049_ = 0f;
                                                } else {
                                                    if (_e3588_ >= 0.999995f) {
                                                        phi_8048_ = 0.999995f;
                                                    } else {
                                                        if (_e3618_ == 2139095040u) {
                                                            phi_8042_ = true;
                                                        } else {
                                                            phi_8042_ = (_e3618_ == 0u);
                                                        }
                                                        let _e3627_ = phi_8042_;
                                                        phi_8048_ = bitcast<f32>(select(_e3617_, 0u, _e3627_));
                                                    }
                                                    let _e3631_ = phi_8048_;
                                                    phi_8049_ = _e3631_;
                                                }
                                                let _e3633_ = phi_8049_;
                                                phi_8056_ = _e3633_;
                                            } else {
                                                phi_8056_ = select(0f, 0.999995f, (_e3617_ == 2139095040u));
                                            }
                                            let _e3635_ = phi_8056_;
                                            let _e3636_ = bitcast<u32>(_e3595_);
                                            let _e3637_ = (_e3636_ & 2139095040u);
                                            if (_e3637_ != 2139095040u) {
                                                if (_e3595_ <= 0f) {
                                                    phi_8088_ = 0f;
                                                } else {
                                                    if (_e3595_ >= 1f) {
                                                        phi_8087_ = 1f;
                                                    } else {
                                                        if (_e3637_ == 2139095040u) {
                                                            phi_8081_ = true;
                                                        } else {
                                                            phi_8081_ = (_e3637_ == 0u);
                                                        }
                                                        let _e3646_ = phi_8081_;
                                                        phi_8087_ = bitcast<f32>(select(_e3636_, 0u, _e3646_));
                                                    }
                                                    let _e3650_ = phi_8087_;
                                                    phi_8088_ = _e3650_;
                                                }
                                                let _e3652_ = phi_8088_;
                                                phi_8095_ = _e3652_;
                                            } else {
                                                phi_8095_ = select(0f, 1f, (_e3636_ == 2139095040u));
                                            }
                                            let _e3654_ = phi_8095_;
                                            let _e3655_ = bitcast<u32>(_e3602_);
                                            let _e3656_ = (_e3655_ & 2139095040u);
                                            if (_e3656_ != 2139095040u) {
                                                if (_e3602_ <= 0f) {
                                                    phi_8127_ = 0f;
                                                } else {
                                                    if (_e3602_ >= 1f) {
                                                        phi_8126_ = 1f;
                                                    } else {
                                                        if (_e3656_ == 2139095040u) {
                                                            phi_8120_ = true;
                                                        } else {
                                                            phi_8120_ = (_e3656_ == 0u);
                                                        }
                                                        let _e3665_ = phi_8120_;
                                                        phi_8126_ = bitcast<f32>(select(_e3655_, 0u, _e3665_));
                                                    }
                                                    let _e3669_ = phi_8126_;
                                                    phi_8127_ = _e3669_;
                                                }
                                                let _e3671_ = phi_8127_;
                                                phi_8134_ = _e3671_;
                                            } else {
                                                phi_8134_ = select(0f, 1f, (_e3655_ == 2139095040u));
                                            }
                                            let _e3673_ = phi_8134_;
                                            let _e3674_ = bitcast<u32>(_e3609_);
                                            let _e3675_ = (_e3674_ & 2139095040u);
                                            if (_e3675_ != 2139095040u) {
                                                if (_e3609_ <= 0f) {
                                                    phi_8166_ = 0f;
                                                } else {
                                                    if (_e3609_ >= 64f) {
                                                        phi_8165_ = 64f;
                                                    } else {
                                                        if (_e3675_ == 2139095040u) {
                                                            phi_8159_ = true;
                                                        } else {
                                                            phi_8159_ = (_e3675_ == 0u);
                                                        }
                                                        let _e3684_ = phi_8159_;
                                                        phi_8165_ = bitcast<f32>(select(_e3674_, 0u, _e3684_));
                                                    }
                                                    let _e3688_ = phi_8165_;
                                                    phi_8166_ = _e3688_;
                                                }
                                                let _e3690_ = phi_8166_;
                                                phi_8173_ = _e3690_;
                                            } else {
                                                phi_8173_ = select(0f, 64f, (_e3674_ == 2139095040u));
                                            }
                                            let _e3692_ = phi_8173_;
                                            let _e3693_ = bitcast<u32>(_e3616_);
                                            let _e3694_ = (_e3693_ & 2139095040u);
                                            if (_e3694_ != 2139095040u) {
                                                if (_e3616_ <= 0f) {
                                                    phi_8205_ = 0f;
                                                } else {
                                                    if (_e3616_ >= 1f) {
                                                        phi_8204_ = 1f;
                                                    } else {
                                                        if (_e3694_ == 2139095040u) {
                                                            phi_8198_ = true;
                                                        } else {
                                                            phi_8198_ = (_e3694_ == 0u);
                                                        }
                                                        let _e3703_ = phi_8198_;
                                                        phi_8204_ = bitcast<f32>(select(_e3693_, 0u, _e3703_));
                                                    }
                                                    let _e3707_ = phi_8204_;
                                                    phi_8205_ = _e3707_;
                                                }
                                                let _e3709_ = phi_8205_;
                                                phi_8212_ = _e3709_;
                                            } else {
                                                phi_8212_ = select(0f, 1f, (_e3693_ == 2139095040u));
                                            }
                                            let _e3711_ = phi_8212_;
                                            switch bitcast<i32>(_e3581_) {
                                                case 0: {
                                                    phi_8307_ = _e418_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8307_ = _e418_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8307_ = _e418_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8307_ = _e418_.w;
                                                    break;
                                                }
                                            }
                                            let _e3718_ = phi_8307_;
                                            switch bitcast<i32>(_e3581_) {
                                                case 0: {
                                                    phi_8318_ = _e425_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8318_ = _e425_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8318_ = _e425_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8318_ = _e425_.w;
                                                    break;
                                                }
                                            }
                                            let _e3725_ = phi_8318_;
                                            if (_e3673_ == 0f) {
                                                phi_8388_ = true;
                                            } else {
                                                phi_8388_ = (_e3711_ >= 1f);
                                            }
                                            let _e3729_ = phi_8388_;
                                            if _e3729_ {
                                                let _e3730_ = select(0u, 2u, _e167_);
                                                let _e3731_ = (_e3730_ + 1u);
                                                if _e1583_ {
                                                    let _e3732_ = (_e3576_.member_1_ * 4u);
                                                    let _e3735_ = (_e164_ & 127u);
                                                    let _e3736_ = (((_e3732_ + _e3730_) * 128u) + _e3735_);
                                                    if (_e3736_ < 2048u) {
                                                    } else {
                                                        phi_34414_ = true;
                                                        break;
                                                    }
                                                    let _e3739_ = lane_buf[_e3736_];
                                                    let _e3742_ = (((_e3732_ + _e3731_) * 128u) + _e3735_);
                                                    if (_e3742_ < 2048u) {
                                                    } else {
                                                        phi_34414_ = true;
                                                        break;
                                                    }
                                                    let _e3745_ = lane_buf[_e3742_];
                                                    let _e3747_ = bitcast<u32>(_e3745_);
                                                    let _e3748_ = (_e3747_ & 2139095040u);
                                                    if (_e3748_ == 2139095040u) {
                                                        phi_8435_ = true;
                                                    } else {
                                                        phi_8435_ = (_e3748_ == 0u);
                                                    }
                                                    let _e3752_ = phi_8435_;
                                                    let _e3756_ = bitcast<u32>((_e3718_ * bitcast<f32>(select(_e3747_, 0u, _e3752_))));
                                                    let _e3757_ = (_e3756_ & 2139095040u);
                                                    if (_e3757_ == 2139095040u) {
                                                        phi_8452_ = true;
                                                    } else {
                                                        phi_8452_ = (_e3757_ == 0u);
                                                    }
                                                    let _e3761_ = phi_8452_;
                                                    if (_e3692_ <= 0f) {
                                                        phi_8464_ = 0f;
                                                    } else {
                                                        phi_8464_ = (_e3692_ * abs(bitcast<f32>(select(_e3756_, 0u, _e3761_))));
                                                    }
                                                    let _e3768_ = phi_8464_;
                                                    let _e3770_ = ((1f - _e3654_) + _e3768_);
                                                    if (_e3770_ <= 0f) {
                                                        phi_8476_ = 0f;
                                                    } else {
                                                        phi_8476_ = select(_e3770_, 1f, (_e3770_ >= 1f));
                                                    }
                                                    let _e3775_ = phi_8476_;
                                                    let _e3776_ = bitcast<u32>((_e3725_ * _e3739_));
                                                    let _e3777_ = (_e3776_ & 2139095040u);
                                                    if (_e3777_ == 2139095040u) {
                                                        phi_8487_ = true;
                                                    } else {
                                                        phi_8487_ = (_e3777_ == 0u);
                                                    }
                                                    let _e3781_ = phi_8487_;
                                                    let _e3783_ = bitcast<f32>(select(_e3776_, 0u, _e3781_));
                                                    let _e3786_ = bitcast<f32>((bitcast<u32>(_e3783_) & 2147483647u));
                                                    if (_e3786_ != _e3786_) {
                                                        phi_24740_ = true;
                                                    } else {
                                                        phi_24740_ = (1000000000000000f <= _e3786_);
                                                    }
                                                    let _e3790_ = phi_24740_;
                                                    let _e3791_ = select(_e3786_, 1000000000000000f, _e3790_);
                                                    let _e3792_ = (_e3791_ * _e3791_);
                                                    let _e3793_ = (_e3792_ + 0.0000010000001f);
                                                    if (_e3793_ != _e3793_) {
                                                        phi_24800_ = true;
                                                    } else {
                                                        phi_24800_ = (0.000000000001f >= _e3793_);
                                                    }
                                                    let _e3797_ = phi_24800_;
                                                    let _e3798_ = select(_e3793_, 0.000000000001f, _e3797_);
                                                    let _e3799_ = (_e3798_ * 0.5f);
                                                    let _e3804_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e3798_) >> bitcast<u32>(1i))));
                                                    let _e3808_ = (_e3804_ * (1.5f - (_e3799_ * (_e3804_ * _e3804_))));
                                                    let _e3812_ = (_e3808_ * (1.5f - (_e3799_ * (_e3808_ * _e3808_))));
                                                    let _e3817_ = (_e3793_ * (_e3812_ * (1.5f - (_e3799_ * (_e3812_ * _e3812_)))));
                                                    let _e3818_ = (_e3817_ + 0.001f);
                                                    if (_e3818_ != _e3818_) {
                                                        phi_24860_ = true;
                                                    } else {
                                                        phi_24860_ = (0.000001f >= _e3818_);
                                                    }
                                                    let _e3822_ = phi_24860_;
                                                    let _e3823_ = select(_e3818_, 0.000001f, _e3822_);
                                                    let _e3826_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3823_)));
                                                    let _e3830_ = (_e3826_ + (_e3826_ * (1f - (_e3823_ * _e3826_))));
                                                    let _e3834_ = (_e3830_ + (_e3830_ * (1f - (_e3823_ * _e3830_))));
                                                    let _e3838_ = (_e3834_ + (_e3834_ * (1f - (_e3823_ * _e3834_))));
                                                    let _e3839_ = (_e3817_ + _e3791_);
                                                    if (_e3839_ != _e3839_) {
                                                        phi_24920_ = true;
                                                    } else {
                                                        phi_24920_ = (0.000001f >= _e3839_);
                                                    }
                                                    let _e3843_ = phi_24920_;
                                                    let _e3844_ = select(_e3839_, 0.000001f, _e3843_);
                                                    let _e3847_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3844_)));
                                                    let _e3851_ = (_e3847_ + (_e3847_ * (1f - (_e3844_ * _e3847_))));
                                                    let _e3855_ = (_e3851_ + (_e3851_ * (1f - (_e3844_ * _e3851_))));
                                                    let _e3865_ = (_e3786_ <= 1000000000000000f);
                                                    let _e3875_ = select(select(select(_e3786_, (0.5f * (_e3791_ + (_e3792_ * _e3838_))), _e3865_), select(-0.0005f, -(((0.0005f * (_e3791_ * (_e3855_ + (_e3855_ * (1f - (_e3844_ * _e3855_)))))) * (1f + (_e3791_ * _e3838_)))), _e3865_), (_e3783_ <= 0f)), 0f, (_e3786_ <= 0f));
                                                    if (_e3783_ <= -1f) {
                                                        phi_8977_ = -1f;
                                                    } else {
                                                        phi_8977_ = select(_e3783_, 1f, (_e3783_ >= 1f));
                                                    }
                                                    let _e3880_ = phi_8977_;
                                                    if (_e3711_ <= 0f) {
                                                        phi_8990_ = _e3875_;
                                                    } else {
                                                        phi_8990_ = (((1f - _e3711_) * _e3875_) + (_e3711_ * (_e3880_ * (1f - (0.33333334f * (_e3880_ * _e3880_))))));
                                                    }
                                                    let _e3891_ = phi_8990_;
                                                    let _e3895_ = bitcast<u32>(((1f - (_e3711_ * _e3775_)) * _e3635_));
                                                    let _e3896_ = (_e3895_ & 2139095040u);
                                                    if (_e3896_ == 2139095040u) {
                                                        phi_9004_ = true;
                                                    } else {
                                                        phi_9004_ = (_e3896_ == 0u);
                                                    }
                                                    let _e3900_ = phi_9004_;
                                                    let _e3904_ = bitcast<u32>((_e3775_ * _e3891_));
                                                    let _e3905_ = (_e3904_ & 2139095040u);
                                                    if (_e3905_ == 2139095040u) {
                                                        phi_9021_ = true;
                                                    } else {
                                                        phi_9021_ = (_e3905_ == 0u);
                                                    }
                                                    let _e3909_ = phi_9021_;
                                                    phi_9029_ = type_28_(bitcast<f32>(select(_e3895_, 0u, _e3900_)), bitcast<f32>(select(_e3904_, 0u, _e3909_)));
                                                } else {
                                                    phi_9029_ = type_28_(1f, 0f);
                                                }
                                                let _e3914_ = phi_9029_;
                                                let _e3917_ = ((16u + select(0u, 4u, _e167_)) + _e3576_.member_1_);
                                                if (_e3917_ < 256u) {
                                                } else {
                                                    phi_34414_ = true;
                                                    break;
                                                }
                                                let _e3920_ = aux_scratch[_e3917_];
                                                let _e3923_ = select(0u, 256u, _e167_);
                                                let _e3924_ = (_e164_ & 127u);
                                                let _e3925_ = (_e3923_ + _e3924_);
                                                let _e3926_ = (_e3925_ < 1536u);
                                                if _e3926_ {
                                                } else {
                                                    phi_34414_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e3925_] = _e3914_.member;
                                                let _e3928_ = (_e3923_ + 128u);
                                                let _e3929_ = (_e3928_ + _e3924_);
                                                let _e3930_ = (_e3929_ < 1536u);
                                                if _e3930_ {
                                                } else {
                                                    phi_34414_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e3929_] = _e3914_.member_1_;
                                                workgroupBarrier();
                                                phi_9070_ = 1u;
                                                phi_9073_ = _e3914_;
                                                loop {
                                                    let _e3933_ = phi_9070_;
                                                    let _e3935_ = phi_9073_;
                                                    local_39_ = _e3935_;
                                                    local_40_ = _e3935_;
                                                    let _e3936_ = (_e3933_ < 128u);
                                                    if _e3936_ {
                                                        if (_e164_ >= _e3933_) {
                                                            let _e3939_ = ((_e164_ - _e3933_) & 127u);
                                                            let _e3940_ = (_e3923_ + _e3939_);
                                                            if (_e3940_ < 1536u) {
                                                            } else {
                                                                phi_34263_ = true;
                                                                break;
                                                            }
                                                            let _e3943_ = scan_scratch[_e3940_];
                                                            let _e3944_ = (_e3928_ + _e3939_);
                                                            if (_e3944_ < 1536u) {
                                                            } else {
                                                                phi_34263_ = true;
                                                                break;
                                                            }
                                                            let _e3947_ = scan_scratch[_e3944_];
                                                            phi_9113_ = type_28_((_e3935_.member * _e3943_), ((_e3935_.member * _e3947_) + _e3935_.member_1_));
                                                        } else {
                                                            phi_9113_ = _e3935_;
                                                        }
                                                        let _e3955_ = phi_9113_;
                                                        workgroupBarrier();
                                                        if _e3926_ {
                                                        } else {
                                                            phi_34263_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e3925_] = _e3955_.member;
                                                        if _e3930_ {
                                                        } else {
                                                            phi_34263_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e3929_] = _e3955_.member_1_;
                                                        workgroupBarrier();
                                                        phi_9071_ = (_e3933_ << bitcast<u32>(1i));
                                                        phi_9074_ = _e3955_;
                                                    } else {
                                                        phi_9071_ = u32();
                                                        phi_9074_ = type_28_();
                                                    }
                                                    let _e3961_ = phi_9071_;
                                                    let _e3963_ = phi_9074_;
                                                    phi_9070_ = _e3961_;
                                                    phi_9073_ = _e3963_;
                                                    phi_34263_ = _e3559_;
                                                    if !(_e3936_) {
                                                        break;
                                                    }
                                                }
                                                let _e3966_ = phi_34263_;
                                                phi_34414_ = _e3966_;
                                                if _e3966_ {
                                                    break;
                                                }
                                                let _e3968_ = local_39_;
                                                let _e3971_ = local_40_;
                                                if _e1582_ {
                                                    if (_e1581_ <= 128u) {
                                                        phi_9149_ = (_e1581_ - 1u);
                                                    } else {
                                                        phi_9149_ = 127u;
                                                    }
                                                    let _e3978_ = phi_9149_;
                                                    let _e3979_ = (_e3978_ & 127u);
                                                    let _e3980_ = (_e3923_ + _e3979_);
                                                    if (_e3980_ < 1536u) {
                                                    } else {
                                                        phi_34414_ = true;
                                                        break;
                                                    }
                                                    let _e3983_ = scan_scratch[_e3980_];
                                                    let _e3984_ = (_e3928_ + _e3979_);
                                                    if (_e3984_ < 1536u) {
                                                    } else {
                                                        phi_34414_ = true;
                                                        break;
                                                    }
                                                    let _e3987_ = scan_scratch[_e3984_];
                                                    phi_9170_ = ((_e3983_ * _e3920_) + _e3987_);
                                                } else {
                                                    phi_9170_ = _e3920_;
                                                }
                                                let _e3991_ = phi_9170_;
                                                workgroupBarrier();
                                                if _e1583_ {
                                                    let _e3992_ = bitcast<u32>(((_e3968_.member * _e3920_) + _e3971_.member_1_));
                                                    let _e3993_ = (_e3992_ & 2139095040u);
                                                    if (_e3993_ == 2139095040u) {
                                                        phi_9184_ = true;
                                                    } else {
                                                        phi_9184_ = (_e3993_ == 0u);
                                                    }
                                                    let _e3997_ = phi_9184_;
                                                    let _e4003_ = ((((_e3576_.member_1_ * 4u) + _e3731_) * 128u) + _e3924_);
                                                    if (_e4003_ < 2048u) {
                                                    } else {
                                                        phi_34414_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e4003_] = bitcast<f32>(select(_e3992_, 0u, _e3997_));
                                                }
                                                if _e1184_ {
                                                    let _e4006_ = bitcast<u32>(_e3991_);
                                                    let _e4007_ = (_e4006_ & 2139095040u);
                                                    if (_e4007_ == 2139095040u) {
                                                        phi_9215_ = true;
                                                    } else {
                                                        phi_9215_ = (_e4007_ == 0u);
                                                    }
                                                    let _e4011_ = phi_9215_;
                                                    let _e4014_ = (16u + _e3576_.member_1_);
                                                    if (_e4014_ < 256u) {
                                                    } else {
                                                        phi_34414_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e4014_] = bitcast<f32>(select(_e4006_, 0u, _e4011_));
                                                }
                                                if (_e163_ == 128u) {
                                                    let _e4018_ = bitcast<u32>(_e3991_);
                                                    let _e4019_ = (_e4018_ & 2139095040u);
                                                    if (_e4019_ == 2139095040u) {
                                                        phi_9242_ = true;
                                                    } else {
                                                        phi_9242_ = (_e4019_ == 0u);
                                                    }
                                                    let _e4023_ = phi_9242_;
                                                    let _e4026_ = (20u + _e3576_.member_1_);
                                                    if (_e4026_ < 256u) {
                                                    } else {
                                                        phi_34414_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e4026_] = bitcast<f32>(select(_e4018_, 0u, _e4023_));
                                                }
                                                workgroupBarrier();
                                                phi_34416_ = _e3966_;
                                            } else {
                                                phi_34416_ = _e3559_;
                                            }
                                            let _e4030_ = phi_34416_;
                                            phi_34415_ = _e4030_;
                                        } else {
                                            phi_34415_ = _e3559_;
                                        }
                                        let _e4032_ = phi_34415_;
                                        phi_34289_ = _e4032_;
                                        phi_7849_ = _e3574_;
                                        phi_34414_ = _e4032_;
                                        if !(_e3580_) {
                                            break;
                                        }
                                    }
                                    let _e4035_ = phi_34414_;
                                    phi_36246_ = _e4035_;
                                    if _e4035_ {
                                        break;
                                    }
                                    if _e203_ {
                                        phi_34610_ = _e4035_;
                                        phi_9260_ = type_15_(0u, 4u);
                                        loop {
                                            let _e4419_ = phi_34610_;
                                            let _e4421_ = phi_9260_;
                                            if (_e4421_.member < _e4421_.member_1_) {
                                                phi_9261_ = type_15_((_e4421_.member + 1u), _e4421_.member_1_);
                                                phi_9284_ = type_15_(1u, _e4421_.member);
                                            } else {
                                                phi_9261_ = _e4421_;
                                                phi_9284_ = type_15_(0u, u32());
                                            }
                                            let _e4434_ = phi_9261_;
                                            let _e4436_ = phi_9284_;
                                            let _e4440_ = (bitcast<i32>(_e4436_.member) != 0i);
                                            if _e4440_ {
                                                let _e4441_ = (_e4436_.member_1_ & 3u);
                                                switch bitcast<i32>(_e4441_) {
                                                    case 0: {
                                                        phi_9302_ = _e376_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9302_ = _e376_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9302_ = _e376_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9302_ = _e376_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4448_ = phi_9302_;
                                                switch bitcast<i32>(_e4441_) {
                                                    case 0: {
                                                        phi_9313_ = _e383_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9313_ = _e383_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9313_ = _e383_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9313_ = _e383_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4455_ = phi_9313_;
                                                switch bitcast<i32>(_e4441_) {
                                                    case 0: {
                                                        phi_9324_ = _e390_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9324_ = _e390_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9324_ = _e390_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9324_ = _e390_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4462_ = phi_9324_;
                                                switch bitcast<i32>(_e4441_) {
                                                    case 0: {
                                                        phi_9335_ = _e397_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9335_ = _e397_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9335_ = _e397_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9335_ = _e397_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4469_ = phi_9335_;
                                                switch bitcast<i32>(_e4441_) {
                                                    case 0: {
                                                        phi_9346_ = _e404_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9346_ = _e404_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9346_ = _e404_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9346_ = _e404_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4476_ = phi_9346_;
                                                let _e4477_ = bitcast<u32>(_e4448_);
                                                let _e4478_ = (_e4477_ & 2139095040u);
                                                if (_e4478_ != 2139095040u) {
                                                    if (_e4448_ <= 0f) {
                                                        phi_9460_ = 0f;
                                                    } else {
                                                        if (_e4448_ >= 0.999995f) {
                                                            phi_9459_ = 0.999995f;
                                                        } else {
                                                            if (_e4478_ == 2139095040u) {
                                                                phi_9453_ = true;
                                                            } else {
                                                                phi_9453_ = (_e4478_ == 0u);
                                                            }
                                                            let _e4487_ = phi_9453_;
                                                            phi_9459_ = bitcast<f32>(select(_e4477_, 0u, _e4487_));
                                                        }
                                                        let _e4491_ = phi_9459_;
                                                        phi_9460_ = _e4491_;
                                                    }
                                                    let _e4493_ = phi_9460_;
                                                    phi_9467_ = _e4493_;
                                                } else {
                                                    phi_9467_ = select(0f, 0.999995f, (_e4477_ == 2139095040u));
                                                }
                                                let _e4495_ = phi_9467_;
                                                let _e4496_ = bitcast<u32>(_e4455_);
                                                let _e4497_ = (_e4496_ & 2139095040u);
                                                if (_e4497_ != 2139095040u) {
                                                    if (_e4455_ <= 0f) {
                                                        phi_9499_ = 0f;
                                                    } else {
                                                        if (_e4455_ >= 1f) {
                                                            phi_9498_ = 1f;
                                                        } else {
                                                            if (_e4497_ == 2139095040u) {
                                                                phi_9492_ = true;
                                                            } else {
                                                                phi_9492_ = (_e4497_ == 0u);
                                                            }
                                                            let _e4506_ = phi_9492_;
                                                            phi_9498_ = bitcast<f32>(select(_e4496_, 0u, _e4506_));
                                                        }
                                                        let _e4510_ = phi_9498_;
                                                        phi_9499_ = _e4510_;
                                                    }
                                                    let _e4512_ = phi_9499_;
                                                    phi_9506_ = _e4512_;
                                                } else {
                                                    phi_9506_ = select(0f, 1f, (_e4496_ == 2139095040u));
                                                }
                                                let _e4514_ = phi_9506_;
                                                let _e4515_ = bitcast<u32>(_e4462_);
                                                let _e4516_ = (_e4515_ & 2139095040u);
                                                if (_e4516_ != 2139095040u) {
                                                    if (_e4462_ <= 0f) {
                                                        phi_9538_ = 0f;
                                                    } else {
                                                        if (_e4462_ >= 1f) {
                                                            phi_9537_ = 1f;
                                                        } else {
                                                            if (_e4516_ == 2139095040u) {
                                                                phi_9531_ = true;
                                                            } else {
                                                                phi_9531_ = (_e4516_ == 0u);
                                                            }
                                                            let _e4525_ = phi_9531_;
                                                            phi_9537_ = bitcast<f32>(select(_e4515_, 0u, _e4525_));
                                                        }
                                                        let _e4529_ = phi_9537_;
                                                        phi_9538_ = _e4529_;
                                                    }
                                                    let _e4531_ = phi_9538_;
                                                    phi_9545_ = _e4531_;
                                                } else {
                                                    phi_9545_ = select(0f, 1f, (_e4515_ == 2139095040u));
                                                }
                                                let _e4533_ = phi_9545_;
                                                let _e4534_ = bitcast<u32>(_e4469_);
                                                let _e4535_ = (_e4534_ & 2139095040u);
                                                if (_e4535_ != 2139095040u) {
                                                    if (_e4469_ <= 0f) {
                                                        phi_9577_ = 0f;
                                                    } else {
                                                        if (_e4469_ >= 64f) {
                                                            phi_9576_ = 64f;
                                                        } else {
                                                            if (_e4535_ == 2139095040u) {
                                                                phi_9570_ = true;
                                                            } else {
                                                                phi_9570_ = (_e4535_ == 0u);
                                                            }
                                                            let _e4544_ = phi_9570_;
                                                            phi_9576_ = bitcast<f32>(select(_e4534_, 0u, _e4544_));
                                                        }
                                                        let _e4548_ = phi_9576_;
                                                        phi_9577_ = _e4548_;
                                                    }
                                                    let _e4550_ = phi_9577_;
                                                    phi_9584_ = _e4550_;
                                                } else {
                                                    phi_9584_ = select(0f, 64f, (_e4534_ == 2139095040u));
                                                }
                                                let _e4552_ = phi_9584_;
                                                let _e4553_ = bitcast<u32>(_e4476_);
                                                let _e4554_ = (_e4553_ & 2139095040u);
                                                if (_e4554_ != 2139095040u) {
                                                    if (_e4476_ <= 0f) {
                                                        phi_9616_ = 0f;
                                                    } else {
                                                        if (_e4476_ >= 1f) {
                                                            phi_9615_ = 1f;
                                                        } else {
                                                            if (_e4554_ == 2139095040u) {
                                                                phi_9609_ = true;
                                                            } else {
                                                                phi_9609_ = (_e4554_ == 0u);
                                                            }
                                                            let _e4563_ = phi_9609_;
                                                            phi_9615_ = bitcast<f32>(select(_e4553_, 0u, _e4563_));
                                                        }
                                                        let _e4567_ = phi_9615_;
                                                        phi_9616_ = _e4567_;
                                                    }
                                                    let _e4569_ = phi_9616_;
                                                    phi_9623_ = _e4569_;
                                                } else {
                                                    phi_9623_ = select(0f, 1f, (_e4553_ == 2139095040u));
                                                }
                                                let _e4571_ = phi_9623_;
                                                switch bitcast<i32>(_e4441_) {
                                                    case 0: {
                                                        phi_9733_ = _e418_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9733_ = _e418_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9733_ = _e418_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9733_ = _e418_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4578_ = phi_9733_;
                                                switch bitcast<i32>(_e4441_) {
                                                    case 0: {
                                                        phi_9744_ = _e425_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9744_ = _e425_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9744_ = _e425_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9744_ = _e425_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4585_ = phi_9744_;
                                                if (_e4533_ == 0f) {
                                                    phi_9814_ = true;
                                                } else {
                                                    phi_9814_ = (_e4571_ >= 1f);
                                                }
                                                let _e4589_ = phi_9814_;
                                                if _e4589_ {
                                                    phi_35365_ = _e4419_;
                                                } else {
                                                    if _e1583_ {
                                                        let _e4595_ = ((((_e4436_.member_1_ * 4u) + select(0u, 2u, _e167_)) * 128u) + (_e164_ & 127u));
                                                        if (_e4595_ < 2048u) {
                                                        } else {
                                                            phi_35363_ = true;
                                                            break;
                                                        }
                                                        let _e4598_ = lane_buf[_e4595_];
                                                        phi_9841_ = _e4598_;
                                                    } else {
                                                        phi_9841_ = 0f;
                                                    }
                                                    let _e4600_ = phi_9841_;
                                                    if _e1583_ {
                                                        let _e4606_ = ((((_e4436_.member_1_ * 4u) + select(1u, 3u, _e167_)) * 128u) + (_e164_ & 127u));
                                                        if (_e4606_ < 2048u) {
                                                        } else {
                                                            phi_35363_ = true;
                                                            break;
                                                        }
                                                        let _e4609_ = lane_buf[_e4606_];
                                                        phi_9861_ = _e4609_;
                                                    } else {
                                                        phi_9861_ = 0f;
                                                    }
                                                    let _e4611_ = phi_9861_;
                                                    let _e4614_ = ((16u + select(0u, 4u, _e167_)) + _e4436_.member_1_);
                                                    if (_e4614_ < 256u) {
                                                    } else {
                                                        phi_35363_ = true;
                                                        break;
                                                    }
                                                    let _e4617_ = aux_scratch[_e4614_];
                                                    let _e4620_ = (select(512u, 640u, _e167_) + _e164_);
                                                    let _e4621_ = (_e4620_ < 1536u);
                                                    if _e4621_ {
                                                    } else {
                                                        phi_35363_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e4620_] = _e4617_;
                                                    workgroupBarrier();
                                                    phi_34582_ = _e4419_;
                                                    phi_9892_ = type_15_(0u, 6u);
                                                    phi_9895_ = _e4617_;
                                                    phi_9897_ = false;
                                                    loop {
                                                        let _e4624_ = phi_34582_;
                                                        let _e4626_ = phi_9892_;
                                                        let _e4628_ = phi_9895_;
                                                        let _e4630_ = phi_9897_;
                                                        local_44_ = _e4628_;
                                                        local_45_ = _e4628_;
                                                        local_46_ = _e4628_;
                                                        local_47_ = _e4628_;
                                                        if (_e4626_.member < _e4626_.member_1_) {
                                                            phi_9893_ = type_15_((_e4626_.member + 1u), _e4626_.member_1_);
                                                            phi_9920_ = type_15_(1u, _e4626_.member);
                                                        } else {
                                                            phi_9893_ = _e4626_;
                                                            phi_9920_ = type_15_(0u, u32());
                                                        }
                                                        let _e4643_ = phi_9893_;
                                                        let _e4645_ = phi_9920_;
                                                        let _e4648_ = (bitcast<i32>(_e4645_.member) != 0i);
                                                        if _e4648_ {
                                                            if _e4630_ {
                                                                phi_35026_ = _e4624_;
                                                                phi_13805_ = _e4628_;
                                                                phi_13806_ = _e4630_;
                                                            } else {
                                                                let _e4649_ = (_e164_ == 0u);
                                                                if _e4649_ {
                                                                    phi_9943_ = _e4617_;
                                                                } else {
                                                                    let _e4650_ = (_e4620_ - 1u);
                                                                    if (_e4650_ < 1536u) {
                                                                    } else {
                                                                        phi_35024_ = true;
                                                                        break;
                                                                    }
                                                                    let _e4653_ = scan_scratch[_e4650_];
                                                                    phi_9943_ = _e4653_;
                                                                }
                                                                let _e4655_ = phi_9943_;
                                                                if _e1583_ {
                                                                    let _e4656_ = (_e4585_ * _e4600_);
                                                                    let _e4658_ = bitcast<u32>((_e4578_ * _e4611_));
                                                                    let _e4659_ = (_e4658_ & 2139095040u);
                                                                    if (_e4659_ == 2139095040u) {
                                                                        phi_9973_ = 0u;
                                                                    } else {
                                                                        if (_e4659_ == 0u) {
                                                                            phi_9972_ = (_e4658_ & 2147483648u);
                                                                        } else {
                                                                            phi_9972_ = _e4658_;
                                                                        }
                                                                        let _e4664_ = phi_9972_;
                                                                        phi_9973_ = _e4664_;
                                                                    }
                                                                    let _e4666_ = phi_9973_;
                                                                    let _e4672_ = (_e4552_ <= 0f);
                                                                    let _e4674_ = (1f - _e4514_);
                                                                    let _e4675_ = (_e4674_ + select((_e4552_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e4666_)) & 2147483647u))), 0f, _e4672_));
                                                                    if (_e4675_ != _e4675_) {
                                                                        phi_25700_ = true;
                                                                    } else {
                                                                        phi_25700_ = (0f >= _e4675_);
                                                                    }
                                                                    let _e4679_ = phi_25700_;
                                                                    let _e4680_ = select(_e4675_, 0f, _e4679_);
                                                                    if (_e4680_ != _e4680_) {
                                                                        phi_25760_ = true;
                                                                    } else {
                                                                        phi_25760_ = (1f <= _e4680_);
                                                                    }
                                                                    let _e4684_ = phi_25760_;
                                                                    let _e4685_ = select(_e4680_, 1f, _e4684_);
                                                                    let _e4686_ = (_e4495_ * _e4655_);
                                                                    let _e4687_ = (1f - _e4571_);
                                                                    let _e4688_ = (_e4533_ * _e4687_);
                                                                    let _e4690_ = (_e4656_ - (_e4688_ * _e4686_));
                                                                    if (_e4690_ != _e4690_) {
                                                                        phi_25820_ = true;
                                                                    } else {
                                                                        phi_25820_ = (-1f >= _e4690_);
                                                                    }
                                                                    let _e4694_ = phi_25820_;
                                                                    let _e4695_ = select(_e4690_, -1f, _e4694_);
                                                                    if (_e4695_ != _e4695_) {
                                                                        phi_25880_ = true;
                                                                    } else {
                                                                        phi_25880_ = (1f <= _e4695_);
                                                                    }
                                                                    let _e4699_ = phi_25880_;
                                                                    let _e4700_ = select(_e4695_, 1f, _e4699_);
                                                                    let _e4707_ = bitcast<f32>((bitcast<u32>(_e4690_) & 2147483647u));
                                                                    if (_e4707_ != _e4707_) {
                                                                        phi_25940_ = true;
                                                                    } else {
                                                                        phi_25940_ = (1000000000000000f <= _e4707_);
                                                                    }
                                                                    let _e4711_ = phi_25940_;
                                                                    let _e4712_ = select(_e4707_, 1000000000000000f, _e4711_);
                                                                    let _e4713_ = (_e4712_ * _e4712_);
                                                                    let _e4714_ = (_e4713_ + 0.0000010000001f);
                                                                    if (_e4714_ != _e4714_) {
                                                                        phi_26000_ = true;
                                                                    } else {
                                                                        phi_26000_ = (0.000000000001f >= _e4714_);
                                                                    }
                                                                    let _e4718_ = phi_26000_;
                                                                    let _e4719_ = select(_e4714_, 0.000000000001f, _e4718_);
                                                                    let _e4720_ = (_e4719_ * 0.5f);
                                                                    let _e4725_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e4719_) >> bitcast<u32>(1i))));
                                                                    let _e4729_ = (_e4725_ * (1.5f - (_e4720_ * (_e4725_ * _e4725_))));
                                                                    let _e4733_ = (_e4729_ * (1.5f - (_e4720_ * (_e4729_ * _e4729_))));
                                                                    let _e4738_ = (_e4714_ * (_e4733_ * (1.5f - (_e4720_ * (_e4733_ * _e4733_)))));
                                                                    let _e4739_ = (_e4738_ + 0.001f);
                                                                    if (_e4739_ != _e4739_) {
                                                                        phi_26060_ = true;
                                                                    } else {
                                                                        phi_26060_ = (0.000001f >= _e4739_);
                                                                    }
                                                                    let _e4743_ = phi_26060_;
                                                                    let _e4744_ = select(_e4739_, 0.000001f, _e4743_);
                                                                    let _e4747_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4744_)));
                                                                    let _e4751_ = (_e4747_ + (_e4747_ * (1f - (_e4744_ * _e4747_))));
                                                                    let _e4755_ = (_e4751_ + (_e4751_ * (1f - (_e4744_ * _e4751_))));
                                                                    let _e4759_ = (_e4755_ + (_e4755_ * (1f - (_e4744_ * _e4755_))));
                                                                    let _e4760_ = (_e4738_ + _e4712_);
                                                                    if (_e4760_ != _e4760_) {
                                                                        phi_26120_ = true;
                                                                    } else {
                                                                        phi_26120_ = (0.000001f >= _e4760_);
                                                                    }
                                                                    let _e4764_ = phi_26120_;
                                                                    let _e4765_ = select(_e4760_, 0.000001f, _e4764_);
                                                                    let _e4768_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4765_)));
                                                                    let _e4772_ = (_e4768_ + (_e4768_ * (1f - (_e4765_ * _e4768_))));
                                                                    let _e4776_ = (_e4772_ + (_e4772_ * (1f - (_e4765_ * _e4772_))));
                                                                    let _e4786_ = (_e4707_ <= 1000000000000000f);
                                                                    let _e4796_ = select(select(select(_e4707_, (0.5f * (_e4712_ + (_e4713_ * _e4759_))), _e4786_), select(-0.0005f, -(((0.0005f * (_e4712_ * (_e4776_ + (_e4776_ * (1f - (_e4765_ * _e4776_)))))) * (1f + (_e4712_ * _e4759_)))), _e4786_), (_e4690_ <= 0f)), 0f, (_e4707_ <= 0f));
                                                                    let _e4800_ = (_e4571_ <= 0f);
                                                                    let _e4807_ = bitcast<u32>((((1f - (_e4571_ * _e4685_)) * _e4686_) + (_e4685_ * select(((_e4687_ * _e4796_) + (_e4571_ * (_e4700_ * (1f - (0.33333334f * (_e4700_ * _e4700_)))))), _e4796_, _e4800_))));
                                                                    let _e4808_ = (_e4807_ & 2139095040u);
                                                                    if (_e4808_ == 2139095040u) {
                                                                        phi_10715_ = 0u;
                                                                    } else {
                                                                        if (_e4808_ == 0u) {
                                                                            phi_10714_ = (_e4807_ & 2147483648u);
                                                                        } else {
                                                                            phi_10714_ = _e4807_;
                                                                        }
                                                                        let _e4813_ = phi_10714_;
                                                                        phi_10715_ = _e4813_;
                                                                    }
                                                                    let _e4815_ = phi_10715_;
                                                                    let _e4817_ = bitcast<u32>(_e4656_);
                                                                    let _e4818_ = (_e4817_ & 2139095040u);
                                                                    if (_e4818_ == 2139095040u) {
                                                                        phi_10759_ = true;
                                                                    } else {
                                                                        phi_10759_ = (_e4818_ == 0u);
                                                                    }
                                                                    let _e4822_ = phi_10759_;
                                                                    let _e4825_ = bitcast<u32>(_e4611_);
                                                                    let _e4826_ = (_e4825_ & 2139095040u);
                                                                    if (_e4826_ == 2139095040u) {
                                                                        phi_10775_ = true;
                                                                    } else {
                                                                        phi_10775_ = (_e4826_ == 0u);
                                                                    }
                                                                    let _e4830_ = phi_10775_;
                                                                    let _e4833_ = bitcast<u32>(_e4655_);
                                                                    let _e4834_ = (_e4833_ & 2139095040u);
                                                                    if (_e4834_ == 2139095040u) {
                                                                        phi_10791_ = true;
                                                                    } else {
                                                                        phi_10791_ = (_e4834_ == 0u);
                                                                    }
                                                                    let _e4838_ = phi_10791_;
                                                                    let _e4842_ = bitcast<u32>((_e4578_ * bitcast<f32>(select(_e4825_, 0u, _e4830_))));
                                                                    let _e4843_ = (_e4842_ & 2139095040u);
                                                                    if (_e4843_ == 2139095040u) {
                                                                        phi_10808_ = true;
                                                                    } else {
                                                                        phi_10808_ = (_e4843_ == 0u);
                                                                    }
                                                                    let _e4847_ = phi_10808_;
                                                                    if _e4672_ {
                                                                        phi_10820_ = 0f;
                                                                    } else {
                                                                        phi_10820_ = (_e4552_ * abs(bitcast<f32>(select(_e4842_, 0u, _e4847_))));
                                                                    }
                                                                    let _e4853_ = phi_10820_;
                                                                    let _e4854_ = (_e4674_ + _e4853_);
                                                                    if (_e4854_ <= 0f) {
                                                                        phi_10832_ = 0f;
                                                                    } else {
                                                                        phi_10832_ = select(_e4854_, 1f, (_e4854_ >= 1f));
                                                                    }
                                                                    let _e4859_ = phi_10832_;
                                                                    let _e4862_ = (bitcast<f32>(select(_e4817_, 0u, _e4822_)) - (_e4688_ * (_e4495_ * bitcast<f32>(select(_e4833_, 0u, _e4838_)))));
                                                                    let _e4865_ = bitcast<f32>((bitcast<u32>(_e4862_) & 2147483647u));
                                                                    if (_e4865_ != _e4865_) {
                                                                        phi_26300_ = true;
                                                                    } else {
                                                                        phi_26300_ = (1000000000000000f <= _e4865_);
                                                                    }
                                                                    let _e4869_ = phi_26300_;
                                                                    let _e4870_ = select(_e4865_, 1000000000000000f, _e4869_);
                                                                    let _e4872_ = ((_e4870_ * _e4870_) + 0.0000010000001f);
                                                                    if (_e4872_ != _e4872_) {
                                                                        phi_26360_ = true;
                                                                    } else {
                                                                        phi_26360_ = (0.000000000001f >= _e4872_);
                                                                    }
                                                                    let _e4876_ = phi_26360_;
                                                                    let _e4877_ = select(_e4872_, 0.000000000001f, _e4876_);
                                                                    let _e4878_ = (_e4877_ * 0.5f);
                                                                    let _e4883_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e4877_) >> bitcast<u32>(1i))));
                                                                    let _e4887_ = (_e4883_ * (1.5f - (_e4878_ * (_e4883_ * _e4883_))));
                                                                    let _e4891_ = (_e4887_ * (1.5f - (_e4878_ * (_e4887_ * _e4887_))));
                                                                    let _e4895_ = (_e4891_ * (1.5f - (_e4878_ * (_e4891_ * _e4891_))));
                                                                    let _e4897_ = ((_e4872_ * _e4895_) + _e4870_);
                                                                    if (_e4897_ != _e4897_) {
                                                                        phi_26480_ = true;
                                                                    } else {
                                                                        phi_26480_ = (0.000001f >= _e4897_);
                                                                    }
                                                                    let _e4901_ = phi_26480_;
                                                                    let _e4902_ = select(_e4897_, 0.000001f, _e4901_);
                                                                    let _e4905_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4902_)));
                                                                    let _e4909_ = (_e4905_ + (_e4905_ * (1f - (_e4902_ * _e4905_))));
                                                                    let _e4913_ = (_e4909_ + (_e4909_ * (1f - (_e4902_ * _e4909_))));
                                                                    let _e4918_ = (_e4865_ <= 1000000000000000f);
                                                                    let _e4929_ = select(select(select(1f, (0.5f * (1f + (_e4870_ * _e4895_))), _e4918_), select(0f, ((0.00000050000006f * _e4895_) * (_e4913_ + (_e4913_ * (1f - (_e4902_ * _e4913_))))), _e4918_), (_e4862_ <= 0f)), 0.5f, (_e4865_ <= 0f));
                                                                    if (_e4929_ != _e4929_) {
                                                                        phi_26540_ = true;
                                                                    } else {
                                                                        phi_26540_ = (0f >= _e4929_);
                                                                    }
                                                                    let _e4933_ = phi_26540_;
                                                                    let _e4934_ = select(_e4929_, 0f, _e4933_);
                                                                    if (_e4934_ != _e4934_) {
                                                                        phi_26600_ = true;
                                                                    } else {
                                                                        phi_26600_ = (1f <= _e4934_);
                                                                    }
                                                                    let _e4938_ = phi_26600_;
                                                                    let _e4939_ = select(_e4934_, 1f, _e4938_);
                                                                    if (abs(_e4862_) < 1f) {
                                                                        phi_11328_ = (1f - (_e4862_ * _e4862_));
                                                                    } else {
                                                                        phi_11328_ = 0f;
                                                                    }
                                                                    let _e4945_ = phi_11328_;
                                                                    if _e4800_ {
                                                                        phi_11336_ = _e4939_;
                                                                    } else {
                                                                        phi_11336_ = ((_e4687_ * _e4939_) + (_e4571_ * _e4945_));
                                                                    }
                                                                    let _e4950_ = phi_11336_;
                                                                    let _e4957_ = (_e4495_ * ((1f - (_e4571_ * _e4859_)) - (((_e4859_ * _e4533_) * _e4687_) * _e4950_)));
                                                                    let _e4958_ = bitcast<u32>(_e4957_);
                                                                    let _e4959_ = (_e4958_ & 2139095040u);
                                                                    let _e4960_ = (_e4959_ == 2139095040u);
                                                                    if _e4960_ {
                                                                        if ((_e4958_ & 2147483647u) > 2139095040u) {
                                                                            phi_11393_ = f32();
                                                                            phi_11394_ = true;
                                                                        } else {
                                                                            let _e4975_ = ((_e4958_ & 2147483648u) == 0u);
                                                                            if _e4975_ {
                                                                                phi_11391_ = 0.999995f;
                                                                            } else {
                                                                                phi_11391_ = f32();
                                                                            }
                                                                            let _e4977_ = phi_11391_;
                                                                            phi_11393_ = _e4977_;
                                                                            phi_11394_ = select(true, false, _e4975_);
                                                                        }
                                                                        let _e4980_ = phi_11393_;
                                                                        let _e4982_ = phi_11394_;
                                                                        if _e4982_ {
                                                                            phi_11414_ = -0.999995f;
                                                                        } else {
                                                                            phi_11414_ = _e4980_;
                                                                        }
                                                                        let _e4984_ = phi_11414_;
                                                                        phi_11473_ = _e4984_;
                                                                    } else {
                                                                        if (_e4957_ < -0.999995f) {
                                                                            phi_11472_ = -0.999995f;
                                                                        } else {
                                                                            if (_e4957_ > 0.999995f) {
                                                                                phi_11471_ = 0.999995f;
                                                                            } else {
                                                                                if _e4960_ {
                                                                                    phi_11465_ = true;
                                                                                } else {
                                                                                    phi_11465_ = (_e4959_ == 0u);
                                                                                }
                                                                                let _e4965_ = phi_11465_;
                                                                                phi_11471_ = bitcast<f32>(select(_e4958_, 0u, _e4965_));
                                                                            }
                                                                            let _e4969_ = phi_11471_;
                                                                            phi_11472_ = _e4969_;
                                                                        }
                                                                        let _e4971_ = phi_11472_;
                                                                        phi_11473_ = _e4971_;
                                                                    }
                                                                    let _e4986_ = phi_11473_;
                                                                    phi_11476_ = type_28_(bitcast<f32>(_e4815_), _e4986_);
                                                                } else {
                                                                    phi_11476_ = type_28_(0f, 1f);
                                                                }
                                                                let _e4989_ = phi_11476_;
                                                                if _e1583_ {
                                                                    if ((bitcast<u32>(_e4989_.member) & 2139095040u) != 2139095040u) {
                                                                        let _e4997_ = ((bitcast<u32>(_e4628_) & 2139095040u) != 2139095040u);
                                                                        if _e4997_ {
                                                                            let _e4998_ = abs(_e4989_.member);
                                                                            let _e4999_ = abs(_e4628_);
                                                                            let _e5001_ = select(_e4999_, _e4998_, (_e4998_ > _e4999_));
                                                                            let _e5003_ = select(1f, _e5001_, (_e5001_ > 1f));
                                                                            if ((bitcast<u32>(_e5003_) & 2139095040u) != 2139095040u) {
                                                                                let _e5009_ = (_e5003_ <= 0f);
                                                                                if _e5009_ {
                                                                                    phi_11600_ = f32();
                                                                                } else {
                                                                                    if (_e5003_ < 0.00001f) {
                                                                                        let _e5041_ = (select(0.000000000000000000000001f, _e5003_, (_e5003_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                        if (_e5041_ != _e5041_) {
                                                                                            phi_26720_ = true;
                                                                                        } else {
                                                                                            phi_26720_ = (0.000001f >= _e5041_);
                                                                                        }
                                                                                        let _e5045_ = phi_26720_;
                                                                                        let _e5046_ = select(_e5041_, 0.000001f, _e5045_);
                                                                                        let _e5049_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5046_)));
                                                                                        let _e5053_ = (_e5049_ + (_e5049_ * (1f - (_e5046_ * _e5049_))));
                                                                                        let _e5057_ = (_e5053_ + (_e5053_ * (1f - (_e5046_ * _e5053_))));
                                                                                        let _e5063_ = bitcast<u32>(((_e5057_ + (_e5057_ * (1f - (_e5046_ * _e5057_)))) * 18446744000000000000f));
                                                                                        let _e5064_ = (_e5063_ & 2139095040u);
                                                                                        if (_e5064_ == 2139095040u) {
                                                                                            phi_11558_ = true;
                                                                                        } else {
                                                                                            phi_11558_ = (_e5064_ == 0u);
                                                                                        }
                                                                                        let _e5068_ = phi_11558_;
                                                                                        phi_11599_ = bitcast<f32>(select(_e5063_, 0u, _e5068_));
                                                                                    } else {
                                                                                        if (_e5003_ != _e5003_) {
                                                                                            phi_26660_ = true;
                                                                                        } else {
                                                                                            phi_26660_ = (0.000001f >= _e5003_);
                                                                                        }
                                                                                        let _e5014_ = phi_26660_;
                                                                                        let _e5015_ = select(_e5003_, 0.000001f, _e5014_);
                                                                                        let _e5018_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5015_)));
                                                                                        let _e5022_ = (_e5018_ + (_e5018_ * (1f - (_e5015_ * _e5018_))));
                                                                                        let _e5026_ = (_e5022_ + (_e5022_ * (1f - (_e5015_ * _e5022_))));
                                                                                        let _e5031_ = bitcast<u32>((_e5026_ + (_e5026_ * (1f - (_e5015_ * _e5026_)))));
                                                                                        let _e5032_ = (_e5031_ & 2139095040u);
                                                                                        if (_e5032_ == 2139095040u) {
                                                                                            phi_11593_ = true;
                                                                                        } else {
                                                                                            phi_11593_ = (_e5032_ == 0u);
                                                                                        }
                                                                                        let _e5036_ = phi_11593_;
                                                                                        phi_11599_ = bitcast<f32>(select(_e5031_, 0u, _e5036_));
                                                                                    }
                                                                                    let _e5072_ = phi_11599_;
                                                                                    phi_11600_ = _e5072_;
                                                                                }
                                                                                let _e5074_ = phi_11600_;
                                                                                phi_11601_ = _e5074_;
                                                                                phi_11602_ = _e5009_;
                                                                            } else {
                                                                                phi_11601_ = f32();
                                                                                phi_11602_ = true;
                                                                            }
                                                                            let _e5076_ = phi_11601_;
                                                                            let _e5078_ = phi_11602_;
                                                                            phi_11608_ = (abs((_e4989_.member - _e4628_)) * select(_e5076_, 0f, _e5078_));
                                                                        } else {
                                                                            phi_11608_ = f32();
                                                                        }
                                                                        let _e5082_ = phi_11608_;
                                                                        phi_11610_ = _e5082_;
                                                                        phi_11611_ = select(true, false, _e4997_);
                                                                    } else {
                                                                        phi_11610_ = f32();
                                                                        phi_11611_ = true;
                                                                    }
                                                                    let _e5085_ = phi_11610_;
                                                                    let _e5087_ = phi_11611_;
                                                                    phi_11616_ = select(_e5085_, 1000000000f, _e5087_);
                                                                } else {
                                                                    phi_11616_ = 0f;
                                                                }
                                                                let _e5090_ = phi_11616_;
                                                                let _e5091_ = (_e163_ < 128u);
                                                                if _e5091_ {
                                                                    let _e5092_ = (32u + _e164_);
                                                                    if (_e5092_ < 256u) {
                                                                    } else {
                                                                        phi_35024_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e5092_] = _e5090_;
                                                                }
                                                                workgroupBarrier();
                                                                let _e5095_ = (_e163_ >= 128u);
                                                                if _e5095_ {
                                                                    let _e5096_ = (32u + _e164_);
                                                                    let _e5097_ = (_e5096_ < 256u);
                                                                    if _e5097_ {
                                                                    } else {
                                                                        phi_35024_ = true;
                                                                        break;
                                                                    }
                                                                    let _e5099_ = aux_scratch[_e5096_];
                                                                    if _e5097_ {
                                                                    } else {
                                                                        phi_35024_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e5096_] = select(_e5099_, _e5090_, (_e5090_ > _e5099_));
                                                                }
                                                                workgroupBarrier();
                                                                phi_11654_ = 64u;
                                                                loop {
                                                                    let _e5103_ = phi_11654_;
                                                                    let _e5104_ = (_e5103_ > 0u);
                                                                    if _e5104_ {
                                                                        if (_e164_ < _e5103_) {
                                                                            if _e5091_ {
                                                                                let _e5106_ = (32u + _e164_);
                                                                                let _e5107_ = (_e5106_ < 256u);
                                                                                if _e5107_ {
                                                                                } else {
                                                                                    phi_34573_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e5109_ = aux_scratch[_e5106_];
                                                                                let _e5110_ = (_e5106_ + _e5103_);
                                                                                if (_e5110_ < 256u) {
                                                                                } else {
                                                                                    phi_34573_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e5113_ = aux_scratch[_e5110_];
                                                                                if _e5107_ {
                                                                                } else {
                                                                                    phi_34573_ = true;
                                                                                    break;
                                                                                }
                                                                                aux_scratch[_e5106_] = select(_e5109_, _e5113_, (_e5113_ > _e5109_));
                                                                            }
                                                                        }
                                                                        workgroupBarrier();
                                                                        phi_11655_ = (_e5103_ >> bitcast<u32>(1i));
                                                                    } else {
                                                                        phi_11655_ = u32();
                                                                    }
                                                                    let _e5119_ = phi_11655_;
                                                                    phi_11654_ = _e5119_;
                                                                    phi_34573_ = _e4624_;
                                                                    if !(_e5104_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e5122_ = phi_34573_;
                                                                phi_35024_ = _e5122_;
                                                                if _e5122_ {
                                                                    break;
                                                                }
                                                                let _e5124_ = aux_scratch[32];
                                                                workgroupBarrier();
                                                                if (_e5124_ <= 0.0000005f) {
                                                                    phi_35027_ = _e5122_;
                                                                    phi_13803_ = _e4628_;
                                                                    phi_13804_ = true;
                                                                } else {
                                                                    if _e1583_ {
                                                                        phi_11714_ = type_28_(_e4989_.member_1_, (_e4989_.member - (_e4989_.member_1_ * _e4655_)));
                                                                    } else {
                                                                        phi_11714_ = type_28_(1f, 0f);
                                                                    }
                                                                    let _e5130_ = phi_11714_;
                                                                    let _e5133_ = select(0u, 256u, _e167_);
                                                                    let _e5134_ = (_e164_ & 127u);
                                                                    let _e5135_ = (_e5133_ + _e5134_);
                                                                    let _e5136_ = (_e5135_ < 1536u);
                                                                    if _e5136_ {
                                                                    } else {
                                                                        phi_35024_ = true;
                                                                        break;
                                                                    }
                                                                    scan_scratch[_e5135_] = _e5130_.member;
                                                                    let _e5138_ = (_e5133_ + 128u);
                                                                    let _e5139_ = (_e5138_ + _e5134_);
                                                                    let _e5140_ = (_e5139_ < 1536u);
                                                                    if _e5140_ {
                                                                    } else {
                                                                        phi_35024_ = true;
                                                                        break;
                                                                    }
                                                                    scan_scratch[_e5139_] = _e5130_.member_1_;
                                                                    workgroupBarrier();
                                                                    phi_11742_ = 1u;
                                                                    phi_11745_ = _e5130_;
                                                                    loop {
                                                                        let _e5143_ = phi_11742_;
                                                                        let _e5145_ = phi_11745_;
                                                                        local_42_ = _e5145_;
                                                                        local_43_ = _e5145_;
                                                                        let _e5146_ = (_e5143_ < 128u);
                                                                        if _e5146_ {
                                                                            if (_e164_ >= _e5143_) {
                                                                                let _e5149_ = ((_e164_ - _e5143_) & 127u);
                                                                                let _e5150_ = (_e5133_ + _e5149_);
                                                                                if (_e5150_ < 1536u) {
                                                                                } else {
                                                                                    phi_34864_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e5153_ = scan_scratch[_e5150_];
                                                                                let _e5154_ = (_e5138_ + _e5149_);
                                                                                if (_e5154_ < 1536u) {
                                                                                } else {
                                                                                    phi_34864_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e5157_ = scan_scratch[_e5154_];
                                                                                phi_11785_ = type_28_((_e5145_.member * _e5153_), ((_e5145_.member * _e5157_) + _e5145_.member_1_));
                                                                            } else {
                                                                                phi_11785_ = _e5145_;
                                                                            }
                                                                            let _e5165_ = phi_11785_;
                                                                            workgroupBarrier();
                                                                            if _e5136_ {
                                                                            } else {
                                                                                phi_34864_ = true;
                                                                                break;
                                                                            }
                                                                            scan_scratch[_e5135_] = _e5165_.member;
                                                                            if _e5140_ {
                                                                            } else {
                                                                                phi_34864_ = true;
                                                                                break;
                                                                            }
                                                                            scan_scratch[_e5139_] = _e5165_.member_1_;
                                                                            workgroupBarrier();
                                                                            phi_11743_ = (_e5143_ << bitcast<u32>(1i));
                                                                            phi_11746_ = _e5165_;
                                                                        } else {
                                                                            phi_11743_ = u32();
                                                                            phi_11746_ = type_28_();
                                                                        }
                                                                        let _e5171_ = phi_11743_;
                                                                        let _e5173_ = phi_11746_;
                                                                        phi_11742_ = _e5171_;
                                                                        phi_11745_ = _e5173_;
                                                                        phi_34864_ = _e5122_;
                                                                        if !(_e5146_) {
                                                                            break;
                                                                        }
                                                                    }
                                                                    let _e5176_ = phi_34864_;
                                                                    phi_35024_ = _e5176_;
                                                                    if _e5176_ {
                                                                        break;
                                                                    }
                                                                    let _e5178_ = local_42_;
                                                                    let _e5181_ = local_43_;
                                                                    if _e1582_ {
                                                                        if (_e1581_ <= 128u) {
                                                                            phi_11821_ = (_e1581_ - 1u);
                                                                        } else {
                                                                            phi_11821_ = 127u;
                                                                        }
                                                                        let _e5188_ = phi_11821_;
                                                                        let _e5189_ = (_e5188_ & 127u);
                                                                        if ((_e5133_ + _e5189_) < 1536u) {
                                                                        } else {
                                                                            phi_35024_ = true;
                                                                            break;
                                                                        }
                                                                        if ((_e5138_ + _e5189_) < 1536u) {
                                                                        } else {
                                                                            phi_35024_ = true;
                                                                            break;
                                                                        }
                                                                    }
                                                                    workgroupBarrier();
                                                                    phi_34879_ = _e5176_;
                                                                    phi_11843_ = type_33_(0i, 3i);
                                                                    phi_11846_ = _e4628_;
                                                                    phi_11848_ = 1f;
                                                                    phi_11850_ = false;
                                                                    phi_11852_ = _e4630_;
                                                                    loop {
                                                                        let _e5196_ = phi_34879_;
                                                                        let _e5198_ = phi_11843_;
                                                                        let _e5200_ = phi_11846_;
                                                                        let _e5202_ = phi_11848_;
                                                                        let _e5204_ = phi_11850_;
                                                                        let _e5206_ = phi_11852_;
                                                                        local_51_ = _e5200_;
                                                                        local_52_ = _e5206_;
                                                                        if (_e5198_.member < _e5198_.member_1_) {
                                                                            let _e5213_ = (_e5198_.member + 1i);
                                                                            if select(false, true, ((false == (_e5213_ > _e5198_.member)) != false)) {
                                                                                phi_11879_ = type_34_(0u, i32());
                                                                            } else {
                                                                                phi_11879_ = type_34_(1u, _e5213_);
                                                                            }
                                                                            let _e5223_ = phi_11879_;
                                                                            if (bitcast<i32>(_e5223_.member) != 0i) {
                                                                            } else {
                                                                                phi_35018_ = true;
                                                                                break;
                                                                            }
                                                                            phi_11844_ = type_33_(_e5223_.member_1_, _e5198_.member_1_);
                                                                            phi_11893_ = type_34_(1u, _e5198_.member);
                                                                        } else {
                                                                            phi_11844_ = _e5198_;
                                                                            phi_11893_ = type_34_(0u, i32());
                                                                        }
                                                                        let _e5233_ = phi_11844_;
                                                                        let _e5235_ = phi_11893_;
                                                                        let _e5239_ = (bitcast<i32>(_e5235_.member) != 0i);
                                                                        if _e5239_ {
                                                                            if _e5204_ {
                                                                                phi_35020_ = _e5196_;
                                                                                phi_13799_ = _e5200_;
                                                                                phi_13800_ = _e5202_;
                                                                                phi_13801_ = _e5204_;
                                                                                phi_13802_ = _e5206_;
                                                                            } else {
                                                                                if _e1583_ {
                                                                                    let _e5241_ = (_e5200_ + (_e5202_ * (((_e5178_.member * _e4617_) + _e5181_.member_1_) - _e4628_)));
                                                                                    let _e5242_ = bitcast<u32>(_e5241_);
                                                                                    let _e5243_ = (_e5242_ & 2139095040u);
                                                                                    let _e5244_ = (_e5243_ == 2139095040u);
                                                                                    if _e5244_ {
                                                                                        if ((_e5242_ & 2147483647u) > 2139095040u) {
                                                                                            phi_11953_ = f32();
                                                                                            phi_11954_ = true;
                                                                                        } else {
                                                                                            let _e5259_ = ((_e5242_ & 2147483648u) == 0u);
                                                                                            if _e5259_ {
                                                                                                phi_11951_ = 1000000f;
                                                                                            } else {
                                                                                                phi_11951_ = f32();
                                                                                            }
                                                                                            let _e5261_ = phi_11951_;
                                                                                            phi_11953_ = _e5261_;
                                                                                            phi_11954_ = select(true, false, _e5259_);
                                                                                        }
                                                                                        let _e5264_ = phi_11953_;
                                                                                        let _e5266_ = phi_11954_;
                                                                                        if _e5266_ {
                                                                                            phi_11974_ = -1000000f;
                                                                                        } else {
                                                                                            phi_11974_ = _e5264_;
                                                                                        }
                                                                                        let _e5268_ = phi_11974_;
                                                                                        phi_12033_ = _e5268_;
                                                                                    } else {
                                                                                        if (_e5241_ < -1000000f) {
                                                                                            phi_12032_ = -1000000f;
                                                                                        } else {
                                                                                            if (_e5241_ > 1000000f) {
                                                                                                phi_12031_ = 1000000f;
                                                                                            } else {
                                                                                                if _e5244_ {
                                                                                                    phi_12025_ = true;
                                                                                                } else {
                                                                                                    phi_12025_ = (_e5243_ == 0u);
                                                                                                }
                                                                                                let _e5249_ = phi_12025_;
                                                                                                phi_12031_ = bitcast<f32>(select(_e5242_, 0u, _e5249_));
                                                                                            }
                                                                                            let _e5253_ = phi_12031_;
                                                                                            phi_12032_ = _e5253_;
                                                                                        }
                                                                                        let _e5255_ = phi_12032_;
                                                                                        phi_12033_ = _e5255_;
                                                                                    }
                                                                                    let _e5270_ = phi_12033_;
                                                                                    phi_12034_ = _e5270_;
                                                                                } else {
                                                                                    phi_12034_ = _e4617_;
                                                                                }
                                                                                let _e5272_ = phi_12034_;
                                                                                let _e5273_ = (select(768u, 896u, _e167_) + _e164_);
                                                                                if (_e5273_ < 1536u) {
                                                                                } else {
                                                                                    phi_35018_ = true;
                                                                                    break;
                                                                                }
                                                                                scan_scratch[_e5273_] = _e5272_;
                                                                                workgroupBarrier();
                                                                                if _e4649_ {
                                                                                    phi_12055_ = _e4617_;
                                                                                } else {
                                                                                    let _e5276_ = (_e5273_ - 1u);
                                                                                    if (_e5276_ < 1536u) {
                                                                                    } else {
                                                                                        phi_35018_ = true;
                                                                                        break;
                                                                                    }
                                                                                    let _e5279_ = scan_scratch[_e5276_];
                                                                                    phi_12055_ = _e5279_;
                                                                                }
                                                                                let _e5281_ = phi_12055_;
                                                                                if _e1583_ {
                                                                                    let _e5284_ = bitcast<u32>((_e4578_ * _e4611_));
                                                                                    let _e5285_ = (_e5284_ & 2139095040u);
                                                                                    if (_e5285_ == 2139095040u) {
                                                                                        phi_12085_ = 0u;
                                                                                    } else {
                                                                                        if (_e5285_ == 0u) {
                                                                                            phi_12084_ = (_e5284_ & 2147483648u);
                                                                                        } else {
                                                                                            phi_12084_ = _e5284_;
                                                                                        }
                                                                                        let _e5290_ = phi_12084_;
                                                                                        phi_12085_ = _e5290_;
                                                                                    }
                                                                                    let _e5292_ = phi_12085_;
                                                                                    let _e5301_ = ((1f - _e4514_) + select((_e4552_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e5292_)) & 2147483647u))), 0f, (_e4552_ <= 0f)));
                                                                                    if (_e5301_ != _e5301_) {
                                                                                        phi_26780_ = true;
                                                                                    } else {
                                                                                        phi_26780_ = (0f >= _e5301_);
                                                                                    }
                                                                                    let _e5305_ = phi_26780_;
                                                                                    let _e5306_ = select(_e5301_, 0f, _e5305_);
                                                                                    if (_e5306_ != _e5306_) {
                                                                                        phi_26840_ = true;
                                                                                    } else {
                                                                                        phi_26840_ = (1f <= _e5306_);
                                                                                    }
                                                                                    let _e5310_ = phi_26840_;
                                                                                    let _e5311_ = select(_e5306_, 1f, _e5310_);
                                                                                    let _e5312_ = (_e4495_ * _e5281_);
                                                                                    let _e5313_ = (1f - _e4571_);
                                                                                    let _e5316_ = ((_e4585_ * _e4600_) - ((_e4533_ * _e5313_) * _e5312_));
                                                                                    if (_e5316_ != _e5316_) {
                                                                                        phi_26900_ = true;
                                                                                    } else {
                                                                                        phi_26900_ = (-1f >= _e5316_);
                                                                                    }
                                                                                    let _e5320_ = phi_26900_;
                                                                                    let _e5321_ = select(_e5316_, -1f, _e5320_);
                                                                                    if (_e5321_ != _e5321_) {
                                                                                        phi_26960_ = true;
                                                                                    } else {
                                                                                        phi_26960_ = (1f <= _e5321_);
                                                                                    }
                                                                                    let _e5325_ = phi_26960_;
                                                                                    let _e5326_ = select(_e5321_, 1f, _e5325_);
                                                                                    let _e5333_ = bitcast<f32>((bitcast<u32>(_e5316_) & 2147483647u));
                                                                                    if (_e5333_ != _e5333_) {
                                                                                        phi_27020_ = true;
                                                                                    } else {
                                                                                        phi_27020_ = (1000000000000000f <= _e5333_);
                                                                                    }
                                                                                    let _e5337_ = phi_27020_;
                                                                                    let _e5338_ = select(_e5333_, 1000000000000000f, _e5337_);
                                                                                    let _e5339_ = (_e5338_ * _e5338_);
                                                                                    let _e5340_ = (_e5339_ + 0.0000010000001f);
                                                                                    if (_e5340_ != _e5340_) {
                                                                                        phi_27080_ = true;
                                                                                    } else {
                                                                                        phi_27080_ = (0.000000000001f >= _e5340_);
                                                                                    }
                                                                                    let _e5344_ = phi_27080_;
                                                                                    let _e5345_ = select(_e5340_, 0.000000000001f, _e5344_);
                                                                                    let _e5346_ = (_e5345_ * 0.5f);
                                                                                    let _e5351_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e5345_) >> bitcast<u32>(1i))));
                                                                                    let _e5355_ = (_e5351_ * (1.5f - (_e5346_ * (_e5351_ * _e5351_))));
                                                                                    let _e5359_ = (_e5355_ * (1.5f - (_e5346_ * (_e5355_ * _e5355_))));
                                                                                    let _e5364_ = (_e5340_ * (_e5359_ * (1.5f - (_e5346_ * (_e5359_ * _e5359_)))));
                                                                                    let _e5365_ = (_e5364_ + 0.001f);
                                                                                    if (_e5365_ != _e5365_) {
                                                                                        phi_27140_ = true;
                                                                                    } else {
                                                                                        phi_27140_ = (0.000001f >= _e5365_);
                                                                                    }
                                                                                    let _e5369_ = phi_27140_;
                                                                                    let _e5370_ = select(_e5365_, 0.000001f, _e5369_);
                                                                                    let _e5373_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5370_)));
                                                                                    let _e5377_ = (_e5373_ + (_e5373_ * (1f - (_e5370_ * _e5373_))));
                                                                                    let _e5381_ = (_e5377_ + (_e5377_ * (1f - (_e5370_ * _e5377_))));
                                                                                    let _e5385_ = (_e5381_ + (_e5381_ * (1f - (_e5370_ * _e5381_))));
                                                                                    let _e5386_ = (_e5364_ + _e5338_);
                                                                                    if (_e5386_ != _e5386_) {
                                                                                        phi_27200_ = true;
                                                                                    } else {
                                                                                        phi_27200_ = (0.000001f >= _e5386_);
                                                                                    }
                                                                                    let _e5390_ = phi_27200_;
                                                                                    let _e5391_ = select(_e5386_, 0.000001f, _e5390_);
                                                                                    let _e5394_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5391_)));
                                                                                    let _e5398_ = (_e5394_ + (_e5394_ * (1f - (_e5391_ * _e5394_))));
                                                                                    let _e5402_ = (_e5398_ + (_e5398_ * (1f - (_e5391_ * _e5398_))));
                                                                                    let _e5412_ = (_e5333_ <= 1000000000000000f);
                                                                                    let _e5422_ = select(select(select(_e5333_, (0.5f * (_e5338_ + (_e5339_ * _e5385_))), _e5412_), select(-0.0005f, -(((0.0005f * (_e5338_ * (_e5402_ + (_e5402_ * (1f - (_e5391_ * _e5402_)))))) * (1f + (_e5338_ * _e5385_)))), _e5412_), (_e5316_ <= 0f)), 0f, (_e5333_ <= 0f));
                                                                                    let _e5433_ = bitcast<u32>((((1f - (_e4571_ * _e5311_)) * _e5312_) + (_e5311_ * select(((_e5313_ * _e5422_) + (_e4571_ * (_e5326_ * (1f - (0.33333334f * (_e5326_ * _e5326_)))))), _e5422_, (_e4571_ <= 0f)))));
                                                                                    let _e5434_ = (_e5433_ & 2139095040u);
                                                                                    if (_e5434_ == 2139095040u) {
                                                                                        phi_12827_ = 0u;
                                                                                    } else {
                                                                                        if (_e5434_ == 0u) {
                                                                                            phi_12826_ = (_e5433_ & 2147483648u);
                                                                                        } else {
                                                                                            phi_12826_ = _e5433_;
                                                                                        }
                                                                                        let _e5439_ = phi_12826_;
                                                                                        phi_12827_ = _e5439_;
                                                                                    }
                                                                                    let _e5441_ = phi_12827_;
                                                                                    let _e5442_ = bitcast<f32>(_e5441_);
                                                                                    if ((bitcast<u32>(_e5442_) & 2139095040u) != 2139095040u) {
                                                                                        let _e5448_ = ((bitcast<u32>(_e5272_) & 2139095040u) != 2139095040u);
                                                                                        if _e5448_ {
                                                                                            let _e5449_ = abs(_e5442_);
                                                                                            let _e5450_ = abs(_e5272_);
                                                                                            let _e5452_ = select(_e5450_, _e5449_, (_e5449_ > _e5450_));
                                                                                            let _e5454_ = select(1f, _e5452_, (_e5452_ > 1f));
                                                                                            if ((bitcast<u32>(_e5454_) & 2139095040u) != 2139095040u) {
                                                                                                let _e5460_ = (_e5454_ <= 0f);
                                                                                                if _e5460_ {
                                                                                                    phi_13668_ = f32();
                                                                                                } else {
                                                                                                    if (_e5454_ < 0.00001f) {
                                                                                                        let _e5492_ = (select(0.000000000000000000000001f, _e5454_, (_e5454_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                                        if (_e5492_ != _e5492_) {
                                                                                                            phi_27800_ = true;
                                                                                                        } else {
                                                                                                            phi_27800_ = (0.000001f >= _e5492_);
                                                                                                        }
                                                                                                        let _e5496_ = phi_27800_;
                                                                                                        let _e5497_ = select(_e5492_, 0.000001f, _e5496_);
                                                                                                        let _e5500_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5497_)));
                                                                                                        let _e5504_ = (_e5500_ + (_e5500_ * (1f - (_e5497_ * _e5500_))));
                                                                                                        let _e5508_ = (_e5504_ + (_e5504_ * (1f - (_e5497_ * _e5504_))));
                                                                                                        let _e5514_ = bitcast<u32>(((_e5508_ + (_e5508_ * (1f - (_e5497_ * _e5508_)))) * 18446744000000000000f));
                                                                                                        let _e5515_ = (_e5514_ & 2139095040u);
                                                                                                        if (_e5515_ == 2139095040u) {
                                                                                                            phi_13626_ = true;
                                                                                                        } else {
                                                                                                            phi_13626_ = (_e5515_ == 0u);
                                                                                                        }
                                                                                                        let _e5519_ = phi_13626_;
                                                                                                        phi_13667_ = bitcast<f32>(select(_e5514_, 0u, _e5519_));
                                                                                                    } else {
                                                                                                        if (_e5454_ != _e5454_) {
                                                                                                            phi_27740_ = true;
                                                                                                        } else {
                                                                                                            phi_27740_ = (0.000001f >= _e5454_);
                                                                                                        }
                                                                                                        let _e5465_ = phi_27740_;
                                                                                                        let _e5466_ = select(_e5454_, 0.000001f, _e5465_);
                                                                                                        let _e5469_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5466_)));
                                                                                                        let _e5473_ = (_e5469_ + (_e5469_ * (1f - (_e5466_ * _e5469_))));
                                                                                                        let _e5477_ = (_e5473_ + (_e5473_ * (1f - (_e5466_ * _e5473_))));
                                                                                                        let _e5482_ = bitcast<u32>((_e5477_ + (_e5477_ * (1f - (_e5466_ * _e5477_)))));
                                                                                                        let _e5483_ = (_e5482_ & 2139095040u);
                                                                                                        if (_e5483_ == 2139095040u) {
                                                                                                            phi_13661_ = true;
                                                                                                        } else {
                                                                                                            phi_13661_ = (_e5483_ == 0u);
                                                                                                        }
                                                                                                        let _e5487_ = phi_13661_;
                                                                                                        phi_13667_ = bitcast<f32>(select(_e5482_, 0u, _e5487_));
                                                                                                    }
                                                                                                    let _e5523_ = phi_13667_;
                                                                                                    phi_13668_ = _e5523_;
                                                                                                }
                                                                                                let _e5525_ = phi_13668_;
                                                                                                phi_13669_ = _e5525_;
                                                                                                phi_13670_ = _e5460_;
                                                                                            } else {
                                                                                                phi_13669_ = f32();
                                                                                                phi_13670_ = true;
                                                                                            }
                                                                                            let _e5527_ = phi_13669_;
                                                                                            let _e5529_ = phi_13670_;
                                                                                            phi_13676_ = (abs((_e5442_ - _e5272_)) * select(_e5527_, 0f, _e5529_));
                                                                                        } else {
                                                                                            phi_13676_ = f32();
                                                                                        }
                                                                                        let _e5533_ = phi_13676_;
                                                                                        phi_13678_ = _e5533_;
                                                                                        phi_13679_ = select(true, false, _e5448_);
                                                                                    } else {
                                                                                        phi_13678_ = f32();
                                                                                        phi_13679_ = true;
                                                                                    }
                                                                                    let _e5536_ = phi_13678_;
                                                                                    let _e5538_ = phi_13679_;
                                                                                    phi_13684_ = select(_e5536_, 1000000000f, _e5538_);
                                                                                } else {
                                                                                    phi_13684_ = 0f;
                                                                                }
                                                                                let _e5541_ = phi_13684_;
                                                                                if _e5091_ {
                                                                                    let _e5542_ = (32u + _e164_);
                                                                                    if (_e5542_ < 256u) {
                                                                                    } else {
                                                                                        phi_35018_ = true;
                                                                                        break;
                                                                                    }
                                                                                    aux_scratch[_e5542_] = _e5541_;
                                                                                }
                                                                                workgroupBarrier();
                                                                                if _e5095_ {
                                                                                    let _e5545_ = (32u + _e164_);
                                                                                    let _e5546_ = (_e5545_ < 256u);
                                                                                    if _e5546_ {
                                                                                    } else {
                                                                                        phi_35018_ = true;
                                                                                        break;
                                                                                    }
                                                                                    let _e5548_ = aux_scratch[_e5545_];
                                                                                    if _e5546_ {
                                                                                    } else {
                                                                                        phi_35018_ = true;
                                                                                        break;
                                                                                    }
                                                                                    aux_scratch[_e5545_] = select(_e5548_, _e5541_, (_e5541_ > _e5548_));
                                                                                }
                                                                                workgroupBarrier();
                                                                                phi_13722_ = 64u;
                                                                                loop {
                                                                                    let _e5552_ = phi_13722_;
                                                                                    let _e5553_ = (_e5552_ > 0u);
                                                                                    if _e5553_ {
                                                                                        if (_e164_ < _e5552_) {
                                                                                            if _e5091_ {
                                                                                                let _e5555_ = (32u + _e164_);
                                                                                                let _e5556_ = (_e5555_ < 256u);
                                                                                                if _e5556_ {
                                                                                                } else {
                                                                                                    phi_34870_ = true;
                                                                                                    break;
                                                                                                }
                                                                                                let _e5558_ = aux_scratch[_e5555_];
                                                                                                let _e5559_ = (_e5555_ + _e5552_);
                                                                                                if (_e5559_ < 256u) {
                                                                                                } else {
                                                                                                    phi_34870_ = true;
                                                                                                    break;
                                                                                                }
                                                                                                let _e5562_ = aux_scratch[_e5559_];
                                                                                                if _e5556_ {
                                                                                                } else {
                                                                                                    phi_34870_ = true;
                                                                                                    break;
                                                                                                }
                                                                                                aux_scratch[_e5555_] = select(_e5558_, _e5562_, (_e5562_ > _e5558_));
                                                                                            }
                                                                                        }
                                                                                        workgroupBarrier();
                                                                                        phi_13723_ = (_e5552_ >> bitcast<u32>(1i));
                                                                                    } else {
                                                                                        phi_13723_ = u32();
                                                                                    }
                                                                                    let _e5568_ = phi_13723_;
                                                                                    phi_13722_ = _e5568_;
                                                                                    phi_34870_ = _e5196_;
                                                                                    if !(_e5553_) {
                                                                                        break;
                                                                                    }
                                                                                }
                                                                                let _e5571_ = phi_34870_;
                                                                                phi_35018_ = _e5571_;
                                                                                if _e5571_ {
                                                                                    break;
                                                                                }
                                                                                let _e5572_ = aux_scratch[32];
                                                                                workgroupBarrier();
                                                                                if (_e5572_ <= _e5124_) {
                                                                                    phi_13779_ = true;
                                                                                } else {
                                                                                    phi_13779_ = (_e5235_.member_1_ == 2i);
                                                                                }
                                                                                let _e5576_ = phi_13779_;
                                                                                if _e5576_ {
                                                                                    if _e4621_ {
                                                                                    } else {
                                                                                        phi_35018_ = true;
                                                                                        break;
                                                                                    }
                                                                                    scan_scratch[_e4620_] = _e5272_;
                                                                                    workgroupBarrier();
                                                                                    phi_13797_ = select(_e5206_, true, (_e5572_ <= 0.0000005f));
                                                                                } else {
                                                                                    phi_13797_ = _e5206_;
                                                                                }
                                                                                let _e5580_ = phi_13797_;
                                                                                phi_35020_ = _e5571_;
                                                                                phi_13799_ = select(_e5200_, _e5272_, _e5576_);
                                                                                phi_13800_ = (_e5202_ * 0.5f);
                                                                                phi_13801_ = select(_e5204_, true, _e5576_);
                                                                                phi_13802_ = _e5580_;
                                                                            }
                                                                            let _e5585_ = phi_35020_;
                                                                            let _e5587_ = phi_13799_;
                                                                            let _e5589_ = phi_13800_;
                                                                            let _e5591_ = phi_13801_;
                                                                            let _e5593_ = phi_13802_;
                                                                            phi_35019_ = _e5585_;
                                                                            phi_11847_ = _e5587_;
                                                                            phi_11849_ = _e5589_;
                                                                            phi_11851_ = _e5591_;
                                                                            phi_11853_ = _e5593_;
                                                                        } else {
                                                                            phi_35019_ = _e5196_;
                                                                            phi_11847_ = f32();
                                                                            phi_11849_ = f32();
                                                                            phi_11851_ = bool();
                                                                            phi_11853_ = bool();
                                                                        }
                                                                        let _e5595_ = phi_35019_;
                                                                        let _e5597_ = phi_11847_;
                                                                        let _e5599_ = phi_11849_;
                                                                        let _e5601_ = phi_11851_;
                                                                        let _e5603_ = phi_11853_;
                                                                        phi_34879_ = _e5595_;
                                                                        phi_11843_ = _e5233_;
                                                                        phi_11846_ = _e5597_;
                                                                        phi_11848_ = _e5599_;
                                                                        phi_11850_ = _e5601_;
                                                                        phi_11852_ = _e5603_;
                                                                        phi_35018_ = _e5595_;
                                                                        if !(_e5239_) {
                                                                            break;
                                                                        }
                                                                    }
                                                                    let _e5606_ = phi_35018_;
                                                                    phi_35024_ = _e5606_;
                                                                    if _e5606_ {
                                                                        break;
                                                                    }
                                                                    phi_35027_ = _e5606_;
                                                                    let _e10098_ = local_51_;
                                                                    phi_13803_ = _e10098_;
                                                                    let _e10101_ = local_52_;
                                                                    phi_13804_ = _e10101_;
                                                                }
                                                                let _e5608_ = phi_35027_;
                                                                let _e5610_ = phi_13803_;
                                                                let _e5612_ = phi_13804_;
                                                                phi_35026_ = _e5608_;
                                                                phi_13805_ = _e5610_;
                                                                phi_13806_ = _e5612_;
                                                            }
                                                            let _e5614_ = phi_35026_;
                                                            let _e5616_ = phi_13805_;
                                                            let _e5618_ = phi_13806_;
                                                            phi_35025_ = _e5614_;
                                                            phi_9896_ = _e5616_;
                                                            phi_9898_ = _e5618_;
                                                        } else {
                                                            phi_35025_ = _e4624_;
                                                            phi_9896_ = f32();
                                                            phi_9898_ = bool();
                                                        }
                                                        let _e5620_ = phi_35025_;
                                                        let _e5622_ = phi_9896_;
                                                        let _e5624_ = phi_9898_;
                                                        phi_34582_ = _e5620_;
                                                        phi_9892_ = _e4643_;
                                                        phi_9895_ = _e5622_;
                                                        phi_9897_ = _e5624_;
                                                        phi_35024_ = _e5620_;
                                                        if !(_e4648_) {
                                                            break;
                                                        }
                                                    }
                                                    let _e5627_ = phi_35024_;
                                                    phi_35363_ = _e5627_;
                                                    if _e5627_ {
                                                        break;
                                                    }
                                                    if (_e164_ == 0u) {
                                                        phi_13820_ = _e4617_;
                                                    } else {
                                                        let _e5629_ = (_e4620_ - 1u);
                                                        if (_e5629_ < 1536u) {
                                                        } else {
                                                            phi_35363_ = true;
                                                            break;
                                                        }
                                                        let _e5632_ = scan_scratch[_e5629_];
                                                        phi_13820_ = _e5632_;
                                                    }
                                                    let _e5634_ = phi_13820_;
                                                    if _e1583_ {
                                                        let _e5638_ = (_e4585_ * _e4600_);
                                                        let _e5640_ = bitcast<u32>((_e4578_ * _e4611_));
                                                        let _e5641_ = (_e5640_ & 2139095040u);
                                                        if (_e5641_ == 2139095040u) {
                                                            phi_13850_ = 0u;
                                                        } else {
                                                            if (_e5641_ == 0u) {
                                                                phi_13849_ = (_e5640_ & 2147483648u);
                                                            } else {
                                                                phi_13849_ = _e5640_;
                                                            }
                                                            let _e5646_ = phi_13849_;
                                                            phi_13850_ = _e5646_;
                                                        }
                                                        let _e5648_ = phi_13850_;
                                                        let _e5654_ = (_e4552_ <= 0f);
                                                        let _e5656_ = (1f - _e4514_);
                                                        let _e5657_ = (_e5656_ + select((_e4552_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e5648_)) & 2147483647u))), 0f, _e5654_));
                                                        if (_e5657_ != _e5657_) {
                                                            phi_27860_ = true;
                                                        } else {
                                                            phi_27860_ = (0f >= _e5657_);
                                                        }
                                                        let _e5661_ = phi_27860_;
                                                        let _e5662_ = select(_e5657_, 0f, _e5661_);
                                                        if (_e5662_ != _e5662_) {
                                                            phi_27920_ = true;
                                                        } else {
                                                            phi_27920_ = (1f <= _e5662_);
                                                        }
                                                        let _e5666_ = phi_27920_;
                                                        let _e5667_ = select(_e5662_, 1f, _e5666_);
                                                        let _e5668_ = (_e4495_ * _e5634_);
                                                        let _e5669_ = (1f - _e4571_);
                                                        let _e5670_ = (_e4533_ * _e5669_);
                                                        let _e5672_ = (_e5638_ - (_e5670_ * _e5668_));
                                                        if (_e5672_ != _e5672_) {
                                                            phi_27980_ = true;
                                                        } else {
                                                            phi_27980_ = (-1f >= _e5672_);
                                                        }
                                                        let _e5676_ = phi_27980_;
                                                        let _e5677_ = select(_e5672_, -1f, _e5676_);
                                                        if (_e5677_ != _e5677_) {
                                                            phi_28040_ = true;
                                                        } else {
                                                            phi_28040_ = (1f <= _e5677_);
                                                        }
                                                        let _e5681_ = phi_28040_;
                                                        let _e5682_ = select(_e5677_, 1f, _e5681_);
                                                        let _e5689_ = bitcast<f32>((bitcast<u32>(_e5672_) & 2147483647u));
                                                        if (_e5689_ != _e5689_) {
                                                            phi_28100_ = true;
                                                        } else {
                                                            phi_28100_ = (1000000000000000f <= _e5689_);
                                                        }
                                                        let _e5693_ = phi_28100_;
                                                        let _e5694_ = select(_e5689_, 1000000000000000f, _e5693_);
                                                        let _e5695_ = (_e5694_ * _e5694_);
                                                        let _e5696_ = (_e5695_ + 0.0000010000001f);
                                                        if (_e5696_ != _e5696_) {
                                                            phi_28160_ = true;
                                                        } else {
                                                            phi_28160_ = (0.000000000001f >= _e5696_);
                                                        }
                                                        let _e5700_ = phi_28160_;
                                                        let _e5701_ = select(_e5696_, 0.000000000001f, _e5700_);
                                                        let _e5702_ = (_e5701_ * 0.5f);
                                                        let _e5707_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e5701_) >> bitcast<u32>(1i))));
                                                        let _e5711_ = (_e5707_ * (1.5f - (_e5702_ * (_e5707_ * _e5707_))));
                                                        let _e5715_ = (_e5711_ * (1.5f - (_e5702_ * (_e5711_ * _e5711_))));
                                                        let _e5720_ = (_e5696_ * (_e5715_ * (1.5f - (_e5702_ * (_e5715_ * _e5715_)))));
                                                        let _e5721_ = (_e5720_ + 0.001f);
                                                        if (_e5721_ != _e5721_) {
                                                            phi_28220_ = true;
                                                        } else {
                                                            phi_28220_ = (0.000001f >= _e5721_);
                                                        }
                                                        let _e5725_ = phi_28220_;
                                                        let _e5726_ = select(_e5721_, 0.000001f, _e5725_);
                                                        let _e5729_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5726_)));
                                                        let _e5733_ = (_e5729_ + (_e5729_ * (1f - (_e5726_ * _e5729_))));
                                                        let _e5737_ = (_e5733_ + (_e5733_ * (1f - (_e5726_ * _e5733_))));
                                                        let _e5741_ = (_e5737_ + (_e5737_ * (1f - (_e5726_ * _e5737_))));
                                                        let _e5742_ = (_e5720_ + _e5694_);
                                                        if (_e5742_ != _e5742_) {
                                                            phi_28280_ = true;
                                                        } else {
                                                            phi_28280_ = (0.000001f >= _e5742_);
                                                        }
                                                        let _e5746_ = phi_28280_;
                                                        let _e5747_ = select(_e5742_, 0.000001f, _e5746_);
                                                        let _e5750_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5747_)));
                                                        let _e5754_ = (_e5750_ + (_e5750_ * (1f - (_e5747_ * _e5750_))));
                                                        let _e5758_ = (_e5754_ + (_e5754_ * (1f - (_e5747_ * _e5754_))));
                                                        let _e5768_ = (_e5689_ <= 1000000000000000f);
                                                        let _e5778_ = select(select(select(_e5689_, (0.5f * (_e5694_ + (_e5695_ * _e5741_))), _e5768_), select(-0.0005f, -(((0.0005f * (_e5694_ * (_e5758_ + (_e5758_ * (1f - (_e5747_ * _e5758_)))))) * (1f + (_e5694_ * _e5741_)))), _e5768_), (_e5672_ <= 0f)), 0f, (_e5689_ <= 0f));
                                                        let _e5782_ = (_e4571_ <= 0f);
                                                        let _e5789_ = bitcast<u32>((((1f - (_e4571_ * _e5667_)) * _e5668_) + (_e5667_ * select(((_e5669_ * _e5778_) + (_e4571_ * (_e5682_ * (1f - (0.33333334f * (_e5682_ * _e5682_)))))), _e5778_, _e5782_))));
                                                        let _e5790_ = (_e5789_ & 2139095040u);
                                                        if (_e5790_ == 2139095040u) {
                                                            phi_14592_ = 0u;
                                                        } else {
                                                            if (_e5790_ == 0u) {
                                                                phi_14591_ = (_e5789_ & 2147483648u);
                                                            } else {
                                                                phi_14591_ = _e5789_;
                                                            }
                                                            let _e5795_ = phi_14591_;
                                                            phi_14592_ = _e5795_;
                                                        }
                                                        let _e5797_ = phi_14592_;
                                                        let _e5799_ = bitcast<u32>(_e5638_);
                                                        let _e5800_ = (_e5799_ & 2139095040u);
                                                        if (_e5800_ == 2139095040u) {
                                                            phi_14636_ = true;
                                                        } else {
                                                            phi_14636_ = (_e5800_ == 0u);
                                                        }
                                                        let _e5804_ = phi_14636_;
                                                        let _e5807_ = bitcast<u32>(_e4611_);
                                                        let _e5808_ = (_e5807_ & 2139095040u);
                                                        if (_e5808_ == 2139095040u) {
                                                            phi_14652_ = true;
                                                        } else {
                                                            phi_14652_ = (_e5808_ == 0u);
                                                        }
                                                        let _e5812_ = phi_14652_;
                                                        let _e5815_ = bitcast<u32>(_e5634_);
                                                        let _e5816_ = (_e5815_ & 2139095040u);
                                                        if (_e5816_ == 2139095040u) {
                                                            phi_14668_ = true;
                                                        } else {
                                                            phi_14668_ = (_e5816_ == 0u);
                                                        }
                                                        let _e5820_ = phi_14668_;
                                                        let _e5824_ = bitcast<u32>((_e4578_ * bitcast<f32>(select(_e5807_, 0u, _e5812_))));
                                                        let _e5825_ = (_e5824_ & 2139095040u);
                                                        if (_e5825_ == 2139095040u) {
                                                            phi_14685_ = true;
                                                        } else {
                                                            phi_14685_ = (_e5825_ == 0u);
                                                        }
                                                        let _e5829_ = phi_14685_;
                                                        if _e5654_ {
                                                            phi_14697_ = 0f;
                                                        } else {
                                                            phi_14697_ = (_e4552_ * abs(bitcast<f32>(select(_e5824_, 0u, _e5829_))));
                                                        }
                                                        let _e5835_ = phi_14697_;
                                                        let _e5836_ = (_e5656_ + _e5835_);
                                                        if (_e5836_ <= 0f) {
                                                            phi_14709_ = 0f;
                                                        } else {
                                                            phi_14709_ = select(_e5836_, 1f, (_e5836_ >= 1f));
                                                        }
                                                        let _e5841_ = phi_14709_;
                                                        let _e5844_ = (bitcast<f32>(select(_e5799_, 0u, _e5804_)) - (_e5670_ * (_e4495_ * bitcast<f32>(select(_e5815_, 0u, _e5820_)))));
                                                        let _e5847_ = bitcast<f32>((bitcast<u32>(_e5844_) & 2147483647u));
                                                        if (_e5847_ != _e5847_) {
                                                            phi_28460_ = true;
                                                        } else {
                                                            phi_28460_ = (1000000000000000f <= _e5847_);
                                                        }
                                                        let _e5851_ = phi_28460_;
                                                        let _e5852_ = select(_e5847_, 1000000000000000f, _e5851_);
                                                        let _e5854_ = ((_e5852_ * _e5852_) + 0.0000010000001f);
                                                        if (_e5854_ != _e5854_) {
                                                            phi_28520_ = true;
                                                        } else {
                                                            phi_28520_ = (0.000000000001f >= _e5854_);
                                                        }
                                                        let _e5858_ = phi_28520_;
                                                        let _e5859_ = select(_e5854_, 0.000000000001f, _e5858_);
                                                        let _e5860_ = (_e5859_ * 0.5f);
                                                        let _e5865_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e5859_) >> bitcast<u32>(1i))));
                                                        let _e5869_ = (_e5865_ * (1.5f - (_e5860_ * (_e5865_ * _e5865_))));
                                                        let _e5873_ = (_e5869_ * (1.5f - (_e5860_ * (_e5869_ * _e5869_))));
                                                        let _e5877_ = (_e5873_ * (1.5f - (_e5860_ * (_e5873_ * _e5873_))));
                                                        let _e5879_ = ((_e5854_ * _e5877_) + _e5852_);
                                                        if (_e5879_ != _e5879_) {
                                                            phi_28640_ = true;
                                                        } else {
                                                            phi_28640_ = (0.000001f >= _e5879_);
                                                        }
                                                        let _e5883_ = phi_28640_;
                                                        let _e5884_ = select(_e5879_, 0.000001f, _e5883_);
                                                        let _e5887_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5884_)));
                                                        let _e5891_ = (_e5887_ + (_e5887_ * (1f - (_e5884_ * _e5887_))));
                                                        let _e5895_ = (_e5891_ + (_e5891_ * (1f - (_e5884_ * _e5891_))));
                                                        let _e5900_ = (_e5847_ <= 1000000000000000f);
                                                        let _e5911_ = select(select(select(1f, (0.5f * (1f + (_e5852_ * _e5877_))), _e5900_), select(0f, ((0.00000050000006f * _e5877_) * (_e5895_ + (_e5895_ * (1f - (_e5884_ * _e5895_))))), _e5900_), (_e5844_ <= 0f)), 0.5f, (_e5847_ <= 0f));
                                                        if (_e5911_ != _e5911_) {
                                                            phi_28700_ = true;
                                                        } else {
                                                            phi_28700_ = (0f >= _e5911_);
                                                        }
                                                        let _e5915_ = phi_28700_;
                                                        let _e5916_ = select(_e5911_, 0f, _e5915_);
                                                        if (_e5916_ != _e5916_) {
                                                            phi_28760_ = true;
                                                        } else {
                                                            phi_28760_ = (1f <= _e5916_);
                                                        }
                                                        let _e5920_ = phi_28760_;
                                                        let _e5921_ = select(_e5916_, 1f, _e5920_);
                                                        if (abs(_e5844_) < 1f) {
                                                            phi_15205_ = (1f - (_e5844_ * _e5844_));
                                                        } else {
                                                            phi_15205_ = 0f;
                                                        }
                                                        let _e5927_ = phi_15205_;
                                                        if _e5782_ {
                                                            phi_15213_ = _e5921_;
                                                        } else {
                                                            phi_15213_ = ((_e5669_ * _e5921_) + (_e4571_ * _e5927_));
                                                        }
                                                        let _e5932_ = phi_15213_;
                                                        let _e5939_ = (_e4495_ * ((1f - (_e4571_ * _e5841_)) - (((_e5841_ * _e4533_) * _e5669_) * _e5932_)));
                                                        let _e5940_ = bitcast<u32>(_e5939_);
                                                        let _e5941_ = (_e5940_ & 2139095040u);
                                                        let _e5942_ = (_e5941_ == 2139095040u);
                                                        if _e5942_ {
                                                            if ((_e5940_ & 2147483647u) > 2139095040u) {
                                                                phi_15270_ = f32();
                                                                phi_15271_ = true;
                                                            } else {
                                                                let _e5957_ = ((_e5940_ & 2147483648u) == 0u);
                                                                if _e5957_ {
                                                                    phi_15268_ = 0.999995f;
                                                                } else {
                                                                    phi_15268_ = f32();
                                                                }
                                                                let _e5959_ = phi_15268_;
                                                                phi_15270_ = _e5959_;
                                                                phi_15271_ = select(true, false, _e5957_);
                                                            }
                                                            let _e5962_ = phi_15270_;
                                                            let _e5964_ = phi_15271_;
                                                            if _e5964_ {
                                                                phi_15291_ = -0.999995f;
                                                            } else {
                                                                phi_15291_ = _e5962_;
                                                            }
                                                            let _e5966_ = phi_15291_;
                                                            phi_15350_ = _e5966_;
                                                        } else {
                                                            if (_e5939_ < -0.999995f) {
                                                                phi_15349_ = -0.999995f;
                                                            } else {
                                                                if (_e5939_ > 0.999995f) {
                                                                    phi_15348_ = 0.999995f;
                                                                } else {
                                                                    if _e5942_ {
                                                                        phi_15342_ = true;
                                                                    } else {
                                                                        phi_15342_ = (_e5941_ == 0u);
                                                                    }
                                                                    let _e5947_ = phi_15342_;
                                                                    phi_15348_ = bitcast<f32>(select(_e5940_, 0u, _e5947_));
                                                                }
                                                                let _e5951_ = phi_15348_;
                                                                phi_15349_ = _e5951_;
                                                            }
                                                            let _e5953_ = phi_15349_;
                                                            phi_15350_ = _e5953_;
                                                        }
                                                        let _e5968_ = phi_15350_;
                                                        phi_15353_ = type_28_(bitcast<f32>(_e5797_), _e5968_);
                                                    } else {
                                                        let _e5636_ = local_44_;
                                                        phi_15353_ = type_28_(_e5636_, 1f);
                                                    }
                                                    let _e5971_ = phi_15353_;
                                                    if _e1583_ {
                                                        if ((bitcast<u32>(_e5971_.member) & 2139095040u) != 2139095040u) {
                                                            let _e5977_ = local_45_;
                                                            let _e5980_ = ((bitcast<u32>(_e5977_) & 2139095040u) != 2139095040u);
                                                            if _e5980_ {
                                                                let _e5981_ = abs(_e5971_.member);
                                                                let _e5983_ = local_46_;
                                                                let _e5984_ = abs(_e5983_);
                                                                let _e5986_ = select(_e5984_, _e5981_, (_e5981_ > _e5984_));
                                                                let _e5988_ = select(1f, _e5986_, (_e5986_ > 1f));
                                                                let _e5990_ = local_47_;
                                                                if ((bitcast<u32>(_e5988_) & 2139095040u) != 2139095040u) {
                                                                    let _e5996_ = (_e5988_ <= 0f);
                                                                    if _e5996_ {
                                                                        phi_15476_ = f32();
                                                                    } else {
                                                                        if (_e5988_ < 0.00001f) {
                                                                            let _e6028_ = (select(0.000000000000000000000001f, _e5988_, (_e5988_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                            if (_e6028_ != _e6028_) {
                                                                                phi_28880_ = true;
                                                                            } else {
                                                                                phi_28880_ = (0.000001f >= _e6028_);
                                                                            }
                                                                            let _e6032_ = phi_28880_;
                                                                            let _e6033_ = select(_e6028_, 0.000001f, _e6032_);
                                                                            let _e6036_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6033_)));
                                                                            let _e6040_ = (_e6036_ + (_e6036_ * (1f - (_e6033_ * _e6036_))));
                                                                            let _e6044_ = (_e6040_ + (_e6040_ * (1f - (_e6033_ * _e6040_))));
                                                                            let _e6050_ = bitcast<u32>(((_e6044_ + (_e6044_ * (1f - (_e6033_ * _e6044_)))) * 18446744000000000000f));
                                                                            let _e6051_ = (_e6050_ & 2139095040u);
                                                                            if (_e6051_ == 2139095040u) {
                                                                                phi_15434_ = true;
                                                                            } else {
                                                                                phi_15434_ = (_e6051_ == 0u);
                                                                            }
                                                                            let _e6055_ = phi_15434_;
                                                                            phi_15475_ = bitcast<f32>(select(_e6050_, 0u, _e6055_));
                                                                        } else {
                                                                            if (_e5988_ != _e5988_) {
                                                                                phi_28820_ = true;
                                                                            } else {
                                                                                phi_28820_ = (0.000001f >= _e5988_);
                                                                            }
                                                                            let _e6001_ = phi_28820_;
                                                                            let _e6002_ = select(_e5988_, 0.000001f, _e6001_);
                                                                            let _e6005_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6002_)));
                                                                            let _e6009_ = (_e6005_ + (_e6005_ * (1f - (_e6002_ * _e6005_))));
                                                                            let _e6013_ = (_e6009_ + (_e6009_ * (1f - (_e6002_ * _e6009_))));
                                                                            let _e6018_ = bitcast<u32>((_e6013_ + (_e6013_ * (1f - (_e6002_ * _e6013_)))));
                                                                            let _e6019_ = (_e6018_ & 2139095040u);
                                                                            if (_e6019_ == 2139095040u) {
                                                                                phi_15469_ = true;
                                                                            } else {
                                                                                phi_15469_ = (_e6019_ == 0u);
                                                                            }
                                                                            let _e6023_ = phi_15469_;
                                                                            phi_15475_ = bitcast<f32>(select(_e6018_, 0u, _e6023_));
                                                                        }
                                                                        let _e6059_ = phi_15475_;
                                                                        phi_15476_ = _e6059_;
                                                                    }
                                                                    let _e6061_ = phi_15476_;
                                                                    phi_15477_ = _e6061_;
                                                                    phi_15478_ = _e5996_;
                                                                } else {
                                                                    phi_15477_ = f32();
                                                                    phi_15478_ = true;
                                                                }
                                                                let _e6063_ = phi_15477_;
                                                                let _e6065_ = phi_15478_;
                                                                phi_15484_ = (abs((_e5971_.member - _e5990_)) * select(_e6063_, 0f, _e6065_));
                                                            } else {
                                                                phi_15484_ = f32();
                                                            }
                                                            let _e6069_ = phi_15484_;
                                                            phi_15486_ = _e6069_;
                                                            phi_15487_ = select(true, false, _e5980_);
                                                        } else {
                                                            phi_15486_ = f32();
                                                            phi_15487_ = true;
                                                        }
                                                        let _e6072_ = phi_15486_;
                                                        let _e6074_ = phi_15487_;
                                                        phi_15492_ = select(_e6072_, 1000000000f, _e6074_);
                                                    } else {
                                                        phi_15492_ = 0f;
                                                    }
                                                    let _e6077_ = phi_15492_;
                                                    let _e6078_ = (_e163_ < 128u);
                                                    if _e6078_ {
                                                        let _e6079_ = (32u + _e164_);
                                                        if (_e6079_ < 256u) {
                                                        } else {
                                                            phi_35363_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e6079_] = _e6077_;
                                                    }
                                                    workgroupBarrier();
                                                    if (_e163_ >= 128u) {
                                                        let _e6083_ = (32u + _e164_);
                                                        let _e6084_ = (_e6083_ < 256u);
                                                        if _e6084_ {
                                                        } else {
                                                            phi_35363_ = true;
                                                            break;
                                                        }
                                                        let _e6086_ = aux_scratch[_e6083_];
                                                        if _e6084_ {
                                                        } else {
                                                            phi_35363_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e6083_] = select(_e6086_, _e6077_, (_e6077_ > _e6086_));
                                                    }
                                                    workgroupBarrier();
                                                    phi_15530_ = 64u;
                                                    loop {
                                                        let _e6090_ = phi_15530_;
                                                        let _e6091_ = (_e6090_ > 0u);
                                                        if _e6091_ {
                                                            if (_e164_ < _e6090_) {
                                                                if _e6078_ {
                                                                    let _e6093_ = (32u + _e164_);
                                                                    let _e6094_ = (_e6093_ < 256u);
                                                                    if _e6094_ {
                                                                    } else {
                                                                        phi_35028_ = true;
                                                                        break;
                                                                    }
                                                                    let _e6096_ = aux_scratch[_e6093_];
                                                                    let _e6097_ = (_e6093_ + _e6090_);
                                                                    if (_e6097_ < 256u) {
                                                                    } else {
                                                                        phi_35028_ = true;
                                                                        break;
                                                                    }
                                                                    let _e6100_ = aux_scratch[_e6097_];
                                                                    if _e6094_ {
                                                                    } else {
                                                                        phi_35028_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e6093_] = select(_e6096_, _e6100_, (_e6100_ > _e6096_));
                                                                }
                                                            }
                                                            workgroupBarrier();
                                                            phi_15531_ = (_e6090_ >> bitcast<u32>(1i));
                                                        } else {
                                                            phi_15531_ = u32();
                                                        }
                                                        let _e6106_ = phi_15531_;
                                                        phi_15530_ = _e6106_;
                                                        phi_35028_ = _e5627_;
                                                        if !(_e6091_) {
                                                            break;
                                                        }
                                                    }
                                                    let _e6109_ = phi_35028_;
                                                    phi_35363_ = _e6109_;
                                                    if _e6109_ {
                                                        break;
                                                    }
                                                    let _e6111_ = aux_scratch[32];
                                                    workgroupBarrier();
                                                    if (_e6111_ > 0.0000005f) {
                                                        if (_e163_ < 2u) {
                                                            let _e6166_ = (_e163_ == 1u);
                                                            let _e6167_ = select(0u, 2u, _e6166_);
                                                            if _e6166_ {
                                                                phi_15610_ = (20u + _e4436_.member_1_);
                                                            } else {
                                                                phi_15610_ = (16u + _e4436_.member_1_);
                                                            }
                                                            let _e6172_ = phi_15610_;
                                                            let _e6173_ = (_e6172_ < 256u);
                                                            if _e6173_ {
                                                            } else {
                                                                phi_35363_ = true;
                                                                break;
                                                            }
                                                            let _e6175_ = aux_scratch[_e6172_];
                                                            phi_15620_ = type_15_(0u, _e1581_);
                                                            phi_15623_ = _e6175_;
                                                            loop {
                                                                let _e6178_ = phi_15620_;
                                                                let _e6180_ = phi_15623_;
                                                                local_48_ = _e6180_;
                                                                if (_e6178_.member < _e6178_.member_1_) {
                                                                    phi_15621_ = type_15_((_e6178_.member + 1u), _e6178_.member_1_);
                                                                    phi_15646_ = type_15_(1u, _e6178_.member);
                                                                } else {
                                                                    phi_15621_ = _e6178_;
                                                                    phi_15646_ = type_15_(0u, u32());
                                                                }
                                                                let _e6193_ = phi_15621_;
                                                                let _e6195_ = phi_15646_;
                                                                let _e6199_ = (bitcast<i32>(_e6195_.member) != 0i);
                                                                if _e6199_ {
                                                                    let _e6201_ = (_e4436_.member_1_ * 4u);
                                                                    let _e6204_ = ((_e6195_.member_1_ & 127u) & 127u);
                                                                    let _e6205_ = (((_e6201_ + _e6167_) * 128u) + _e6204_);
                                                                    if (_e6205_ < 2048u) {
                                                                    } else {
                                                                        phi_35263_ = true;
                                                                        break;
                                                                    }
                                                                    let _e6208_ = lane_buf[_e6205_];
                                                                    let _e6211_ = (((_e6201_ + (_e6167_ + 1u)) * 128u) + _e6204_);
                                                                    let _e6212_ = (_e6211_ < 2048u);
                                                                    if _e6212_ {
                                                                    } else {
                                                                        phi_35263_ = true;
                                                                        break;
                                                                    }
                                                                    let _e6214_ = lane_buf[_e6211_];
                                                                    let _e6217_ = bitcast<u32>((_e4578_ * _e6214_));
                                                                    let _e6218_ = (_e6217_ & 2139095040u);
                                                                    if (_e6218_ == 2139095040u) {
                                                                        phi_15693_ = 0u;
                                                                    } else {
                                                                        if (_e6218_ == 0u) {
                                                                            phi_15692_ = (_e6217_ & 2147483648u);
                                                                        } else {
                                                                            phi_15692_ = _e6217_;
                                                                        }
                                                                        let _e6223_ = phi_15692_;
                                                                        phi_15693_ = _e6223_;
                                                                    }
                                                                    let _e6225_ = phi_15693_;
                                                                    let _e6234_ = ((1f - _e4514_) + select((_e4552_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e6225_)) & 2147483647u))), 0f, (_e4552_ <= 0f)));
                                                                    if (_e6234_ != _e6234_) {
                                                                        phi_28940_ = true;
                                                                    } else {
                                                                        phi_28940_ = (0f >= _e6234_);
                                                                    }
                                                                    let _e6238_ = phi_28940_;
                                                                    let _e6239_ = select(_e6234_, 0f, _e6238_);
                                                                    if (_e6239_ != _e6239_) {
                                                                        phi_29000_ = true;
                                                                    } else {
                                                                        phi_29000_ = (1f <= _e6239_);
                                                                    }
                                                                    let _e6243_ = phi_29000_;
                                                                    let _e6244_ = select(_e6239_, 1f, _e6243_);
                                                                    let _e6245_ = (_e4495_ * _e6180_);
                                                                    let _e6246_ = (1f - _e4571_);
                                                                    let _e6249_ = ((_e4585_ * _e6208_) - ((_e4533_ * _e6246_) * _e6245_));
                                                                    if (_e6249_ != _e6249_) {
                                                                        phi_29060_ = true;
                                                                    } else {
                                                                        phi_29060_ = (-1f >= _e6249_);
                                                                    }
                                                                    let _e6253_ = phi_29060_;
                                                                    let _e6254_ = select(_e6249_, -1f, _e6253_);
                                                                    if (_e6254_ != _e6254_) {
                                                                        phi_29120_ = true;
                                                                    } else {
                                                                        phi_29120_ = (1f <= _e6254_);
                                                                    }
                                                                    let _e6258_ = phi_29120_;
                                                                    let _e6259_ = select(_e6254_, 1f, _e6258_);
                                                                    let _e6266_ = bitcast<f32>((bitcast<u32>(_e6249_) & 2147483647u));
                                                                    if (_e6266_ != _e6266_) {
                                                                        phi_29180_ = true;
                                                                    } else {
                                                                        phi_29180_ = (1000000000000000f <= _e6266_);
                                                                    }
                                                                    let _e6270_ = phi_29180_;
                                                                    let _e6271_ = select(_e6266_, 1000000000000000f, _e6270_);
                                                                    let _e6272_ = (_e6271_ * _e6271_);
                                                                    let _e6273_ = (_e6272_ + 0.0000010000001f);
                                                                    if (_e6273_ != _e6273_) {
                                                                        phi_29240_ = true;
                                                                    } else {
                                                                        phi_29240_ = (0.000000000001f >= _e6273_);
                                                                    }
                                                                    let _e6277_ = phi_29240_;
                                                                    let _e6278_ = select(_e6273_, 0.000000000001f, _e6277_);
                                                                    let _e6279_ = (_e6278_ * 0.5f);
                                                                    let _e6284_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e6278_) >> bitcast<u32>(1i))));
                                                                    let _e6288_ = (_e6284_ * (1.5f - (_e6279_ * (_e6284_ * _e6284_))));
                                                                    let _e6292_ = (_e6288_ * (1.5f - (_e6279_ * (_e6288_ * _e6288_))));
                                                                    let _e6297_ = (_e6273_ * (_e6292_ * (1.5f - (_e6279_ * (_e6292_ * _e6292_)))));
                                                                    let _e6298_ = (_e6297_ + 0.001f);
                                                                    if (_e6298_ != _e6298_) {
                                                                        phi_29300_ = true;
                                                                    } else {
                                                                        phi_29300_ = (0.000001f >= _e6298_);
                                                                    }
                                                                    let _e6302_ = phi_29300_;
                                                                    let _e6303_ = select(_e6298_, 0.000001f, _e6302_);
                                                                    let _e6306_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6303_)));
                                                                    let _e6310_ = (_e6306_ + (_e6306_ * (1f - (_e6303_ * _e6306_))));
                                                                    let _e6314_ = (_e6310_ + (_e6310_ * (1f - (_e6303_ * _e6310_))));
                                                                    let _e6318_ = (_e6314_ + (_e6314_ * (1f - (_e6303_ * _e6314_))));
                                                                    let _e6319_ = (_e6297_ + _e6271_);
                                                                    if (_e6319_ != _e6319_) {
                                                                        phi_29360_ = true;
                                                                    } else {
                                                                        phi_29360_ = (0.000001f >= _e6319_);
                                                                    }
                                                                    let _e6323_ = phi_29360_;
                                                                    let _e6324_ = select(_e6319_, 0.000001f, _e6323_);
                                                                    let _e6327_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6324_)));
                                                                    let _e6331_ = (_e6327_ + (_e6327_ * (1f - (_e6324_ * _e6327_))));
                                                                    let _e6335_ = (_e6331_ + (_e6331_ * (1f - (_e6324_ * _e6331_))));
                                                                    let _e6345_ = (_e6266_ <= 1000000000000000f);
                                                                    let _e6355_ = select(select(select(_e6266_, (0.5f * (_e6271_ + (_e6272_ * _e6318_))), _e6345_), select(-0.0005f, -(((0.0005f * (_e6271_ * (_e6335_ + (_e6335_ * (1f - (_e6324_ * _e6335_)))))) * (1f + (_e6271_ * _e6318_)))), _e6345_), (_e6249_ <= 0f)), 0f, (_e6266_ <= 0f));
                                                                    let _e6366_ = bitcast<u32>((((1f - (_e4571_ * _e6244_)) * _e6245_) + (_e6244_ * select(((_e6246_ * _e6355_) + (_e4571_ * (_e6259_ * (1f - (0.33333334f * (_e6259_ * _e6259_)))))), _e6355_, (_e4571_ <= 0f)))));
                                                                    let _e6367_ = (_e6366_ & 2139095040u);
                                                                    if (_e6367_ == 2139095040u) {
                                                                        phi_16435_ = 0u;
                                                                    } else {
                                                                        if (_e6367_ == 0u) {
                                                                            phi_16434_ = (_e6366_ & 2147483648u);
                                                                        } else {
                                                                            phi_16434_ = _e6366_;
                                                                        }
                                                                        let _e6372_ = phi_16434_;
                                                                        phi_16435_ = _e6372_;
                                                                    }
                                                                    let _e6374_ = phi_16435_;
                                                                    let _e6375_ = bitcast<f32>(_e6374_);
                                                                    if _e6212_ {
                                                                    } else {
                                                                        phi_35263_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e6211_] = _e6375_;
                                                                    phi_15624_ = _e6375_;
                                                                } else {
                                                                    phi_15624_ = f32();
                                                                }
                                                                let _e6377_ = phi_15624_;
                                                                phi_15620_ = _e6193_;
                                                                phi_15623_ = _e6377_;
                                                                phi_35263_ = _e6109_;
                                                                if !(_e6199_) {
                                                                    break;
                                                                }
                                                            }
                                                            let _e6380_ = phi_35263_;
                                                            phi_35363_ = _e6380_;
                                                            if _e6380_ {
                                                                break;
                                                            }
                                                            let _e6382_ = local_48_;
                                                            let _e6383_ = bitcast<u32>(_e6382_);
                                                            let _e6384_ = (_e6383_ & 2139095040u);
                                                            if (_e6384_ == 2139095040u) {
                                                                phi_16489_ = true;
                                                            } else {
                                                                phi_16489_ = (_e6384_ == 0u);
                                                            }
                                                            let _e6388_ = phi_16489_;
                                                            if _e6173_ {
                                                            } else {
                                                                phi_35363_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e6172_] = bitcast<f32>(select(_e6383_, 0u, _e6388_));
                                                            phi_35375_ = _e6380_;
                                                        } else {
                                                            phi_35375_ = _e6109_;
                                                        }
                                                        let _e6392_ = phi_35375_;
                                                        workgroupBarrier();
                                                        phi_35366_ = _e6392_;
                                                    } else {
                                                        workgroupBarrier();
                                                        if _e6078_ {
                                                            if _e1583_ {
                                                                let _e6113_ = (512u + _e164_);
                                                                if (_e6113_ < 1536u) {
                                                                } else {
                                                                    phi_35363_ = true;
                                                                    break;
                                                                }
                                                                let _e6116_ = scan_scratch[_e6113_];
                                                                let _e6117_ = (_e4436_.member_1_ * 4u);
                                                                let _e6120_ = (_e164_ & 127u);
                                                                let _e6121_ = (((_e6117_ + 1u) * 128u) + _e6120_);
                                                                if (_e6121_ < 2048u) {
                                                                } else {
                                                                    phi_35363_ = true;
                                                                    break;
                                                                }
                                                                lane_buf[_e6121_] = _e6116_;
                                                                let _e6124_ = (640u + _e164_);
                                                                if (_e6124_ < 1536u) {
                                                                } else {
                                                                    phi_35363_ = true;
                                                                    break;
                                                                }
                                                                let _e6127_ = scan_scratch[_e6124_];
                                                                let _e6130_ = (((_e6117_ + 3u) * 128u) + _e6120_);
                                                                if (_e6130_ < 2048u) {
                                                                } else {
                                                                    phi_35363_ = true;
                                                                    break;
                                                                }
                                                                lane_buf[_e6130_] = _e6127_;
                                                            }
                                                        }
                                                        if _e1184_ {
                                                            if _e1582_ {
                                                                let _e6134_ = ((_e1581_ - 1u) & 127u);
                                                                let _e6135_ = (512u + _e6134_);
                                                                if (_e6135_ < 1536u) {
                                                                } else {
                                                                    phi_35363_ = true;
                                                                    break;
                                                                }
                                                                let _e6138_ = scan_scratch[_e6135_];
                                                                let _e6139_ = bitcast<u32>(_e6138_);
                                                                let _e6140_ = (_e6139_ & 2139095040u);
                                                                if (_e6140_ == 2139095040u) {
                                                                    phi_16574_ = true;
                                                                } else {
                                                                    phi_16574_ = (_e6140_ == 0u);
                                                                }
                                                                let _e6144_ = phi_16574_;
                                                                let _e6147_ = (16u + _e4436_.member_1_);
                                                                if (_e6147_ < 256u) {
                                                                } else {
                                                                    phi_35363_ = true;
                                                                    break;
                                                                }
                                                                aux_scratch[_e6147_] = bitcast<f32>(select(_e6139_, 0u, _e6144_));
                                                                let _e6150_ = (640u + _e6134_);
                                                                if (_e6150_ < 1536u) {
                                                                } else {
                                                                    phi_35363_ = true;
                                                                    break;
                                                                }
                                                                let _e6153_ = scan_scratch[_e6150_];
                                                                let _e6154_ = bitcast<u32>(_e6153_);
                                                                let _e6155_ = (_e6154_ & 2139095040u);
                                                                if (_e6155_ == 2139095040u) {
                                                                    phi_16605_ = true;
                                                                } else {
                                                                    phi_16605_ = (_e6155_ == 0u);
                                                                }
                                                                let _e6159_ = phi_16605_;
                                                                let _e6162_ = (20u + _e4436_.member_1_);
                                                                if (_e6162_ < 256u) {
                                                                } else {
                                                                    phi_35363_ = true;
                                                                    break;
                                                                }
                                                                aux_scratch[_e6162_] = bitcast<f32>(select(_e6154_, 0u, _e6159_));
                                                            }
                                                        }
                                                        workgroupBarrier();
                                                        phi_35366_ = _e6109_;
                                                    }
                                                    let _e6394_ = phi_35366_;
                                                    phi_35365_ = _e6394_;
                                                }
                                                let _e6396_ = phi_35365_;
                                                phi_35364_ = _e6396_;
                                            } else {
                                                phi_35364_ = _e4419_;
                                            }
                                            let _e6398_ = phi_35364_;
                                            phi_34610_ = _e6398_;
                                            phi_9260_ = _e4434_;
                                            phi_35363_ = _e6398_;
                                            if !(_e4440_) {
                                                break;
                                            }
                                        }
                                        let _e6401_ = phi_35363_;
                                        phi_36246_ = _e6401_;
                                        if _e6401_ {
                                            break;
                                        }
                                        phi_35382_ = _e6401_;
                                    } else {
                                        if (_e163_ < 8u) {
                                            let _e4038_ = (_e163_ >> bitcast<u32>(1i));
                                            let _e4040_ = ((_e163_ & 1u) != 0u);
                                            let _e4041_ = (_e4038_ & 3u);
                                            switch bitcast<i32>(_e4041_) {
                                                case 0: {
                                                    phi_16635_ = _e376_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16635_ = _e376_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16635_ = _e376_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16635_ = _e376_.w;
                                                    break;
                                                }
                                            }
                                            let _e4048_ = phi_16635_;
                                            switch bitcast<i32>(_e4041_) {
                                                case 0: {
                                                    phi_16646_ = _e383_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16646_ = _e383_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16646_ = _e383_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16646_ = _e383_.w;
                                                    break;
                                                }
                                            }
                                            let _e4055_ = phi_16646_;
                                            switch bitcast<i32>(_e4041_) {
                                                case 0: {
                                                    phi_16657_ = _e390_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16657_ = _e390_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16657_ = _e390_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16657_ = _e390_.w;
                                                    break;
                                                }
                                            }
                                            let _e4062_ = phi_16657_;
                                            switch bitcast<i32>(_e4041_) {
                                                case 0: {
                                                    phi_16668_ = _e397_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16668_ = _e397_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16668_ = _e397_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16668_ = _e397_.w;
                                                    break;
                                                }
                                            }
                                            let _e4069_ = phi_16668_;
                                            switch bitcast<i32>(_e4041_) {
                                                case 0: {
                                                    phi_16679_ = _e404_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16679_ = _e404_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16679_ = _e404_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16679_ = _e404_.w;
                                                    break;
                                                }
                                            }
                                            let _e4076_ = phi_16679_;
                                            let _e4077_ = bitcast<u32>(_e4048_);
                                            let _e4078_ = (_e4077_ & 2139095040u);
                                            if (_e4078_ != 2139095040u) {
                                                if (_e4048_ <= 0f) {
                                                    phi_16793_ = 0f;
                                                } else {
                                                    if (_e4048_ >= 0.999995f) {
                                                        phi_16792_ = 0.999995f;
                                                    } else {
                                                        if (_e4078_ == 2139095040u) {
                                                            phi_16786_ = true;
                                                        } else {
                                                            phi_16786_ = (_e4078_ == 0u);
                                                        }
                                                        let _e4087_ = phi_16786_;
                                                        phi_16792_ = bitcast<f32>(select(_e4077_, 0u, _e4087_));
                                                    }
                                                    let _e4091_ = phi_16792_;
                                                    phi_16793_ = _e4091_;
                                                }
                                                let _e4093_ = phi_16793_;
                                                phi_16800_ = _e4093_;
                                            } else {
                                                phi_16800_ = select(0f, 0.999995f, (_e4077_ == 2139095040u));
                                            }
                                            let _e4095_ = phi_16800_;
                                            let _e4096_ = bitcast<u32>(_e4055_);
                                            let _e4097_ = (_e4096_ & 2139095040u);
                                            if (_e4097_ != 2139095040u) {
                                                if (_e4055_ <= 0f) {
                                                    phi_16832_ = 0f;
                                                } else {
                                                    if (_e4055_ >= 1f) {
                                                        phi_16831_ = 1f;
                                                    } else {
                                                        if (_e4097_ == 2139095040u) {
                                                            phi_16825_ = true;
                                                        } else {
                                                            phi_16825_ = (_e4097_ == 0u);
                                                        }
                                                        let _e4106_ = phi_16825_;
                                                        phi_16831_ = bitcast<f32>(select(_e4096_, 0u, _e4106_));
                                                    }
                                                    let _e4110_ = phi_16831_;
                                                    phi_16832_ = _e4110_;
                                                }
                                                let _e4112_ = phi_16832_;
                                                phi_16839_ = _e4112_;
                                            } else {
                                                phi_16839_ = select(0f, 1f, (_e4096_ == 2139095040u));
                                            }
                                            let _e4114_ = phi_16839_;
                                            let _e4115_ = bitcast<u32>(_e4062_);
                                            let _e4116_ = (_e4115_ & 2139095040u);
                                            if (_e4116_ != 2139095040u) {
                                                if (_e4062_ <= 0f) {
                                                    phi_16871_ = 0f;
                                                } else {
                                                    if (_e4062_ >= 1f) {
                                                        phi_16870_ = 1f;
                                                    } else {
                                                        if (_e4116_ == 2139095040u) {
                                                            phi_16864_ = true;
                                                        } else {
                                                            phi_16864_ = (_e4116_ == 0u);
                                                        }
                                                        let _e4125_ = phi_16864_;
                                                        phi_16870_ = bitcast<f32>(select(_e4115_, 0u, _e4125_));
                                                    }
                                                    let _e4129_ = phi_16870_;
                                                    phi_16871_ = _e4129_;
                                                }
                                                let _e4131_ = phi_16871_;
                                                phi_16878_ = _e4131_;
                                            } else {
                                                phi_16878_ = select(0f, 1f, (_e4115_ == 2139095040u));
                                            }
                                            let _e4133_ = phi_16878_;
                                            let _e4134_ = bitcast<u32>(_e4069_);
                                            let _e4135_ = (_e4134_ & 2139095040u);
                                            if (_e4135_ != 2139095040u) {
                                                if (_e4069_ <= 0f) {
                                                    phi_16910_ = 0f;
                                                } else {
                                                    if (_e4069_ >= 64f) {
                                                        phi_16909_ = 64f;
                                                    } else {
                                                        if (_e4135_ == 2139095040u) {
                                                            phi_16903_ = true;
                                                        } else {
                                                            phi_16903_ = (_e4135_ == 0u);
                                                        }
                                                        let _e4144_ = phi_16903_;
                                                        phi_16909_ = bitcast<f32>(select(_e4134_, 0u, _e4144_));
                                                    }
                                                    let _e4148_ = phi_16909_;
                                                    phi_16910_ = _e4148_;
                                                }
                                                let _e4150_ = phi_16910_;
                                                phi_16917_ = _e4150_;
                                            } else {
                                                phi_16917_ = select(0f, 64f, (_e4134_ == 2139095040u));
                                            }
                                            let _e4152_ = phi_16917_;
                                            let _e4153_ = bitcast<u32>(_e4076_);
                                            let _e4154_ = (_e4153_ & 2139095040u);
                                            if (_e4154_ != 2139095040u) {
                                                if (_e4076_ <= 0f) {
                                                    phi_16949_ = 0f;
                                                } else {
                                                    if (_e4076_ >= 1f) {
                                                        phi_16948_ = 1f;
                                                    } else {
                                                        if (_e4154_ == 2139095040u) {
                                                            phi_16942_ = true;
                                                        } else {
                                                            phi_16942_ = (_e4154_ == 0u);
                                                        }
                                                        let _e4163_ = phi_16942_;
                                                        phi_16948_ = bitcast<f32>(select(_e4153_, 0u, _e4163_));
                                                    }
                                                    let _e4167_ = phi_16948_;
                                                    phi_16949_ = _e4167_;
                                                }
                                                let _e4169_ = phi_16949_;
                                                phi_16956_ = _e4169_;
                                            } else {
                                                phi_16956_ = select(0f, 1f, (_e4153_ == 2139095040u));
                                            }
                                            let _e4171_ = phi_16956_;
                                            switch bitcast<i32>(_e4041_) {
                                                case 0: {
                                                    phi_17066_ = _e418_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_17066_ = _e418_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_17066_ = _e418_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_17066_ = _e418_.w;
                                                    break;
                                                }
                                            }
                                            let _e4178_ = phi_17066_;
                                            switch bitcast<i32>(_e4041_) {
                                                case 0: {
                                                    phi_17077_ = _e425_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_17077_ = _e425_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_17077_ = _e425_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_17077_ = _e425_.w;
                                                    break;
                                                }
                                            }
                                            let _e4185_ = phi_17077_;
                                            if (_e4133_ == 0f) {
                                                phi_17147_ = true;
                                            } else {
                                                phi_17147_ = (_e4171_ >= 1f);
                                            }
                                            let _e4189_ = phi_17147_;
                                            if _e4189_ {
                                                phi_35384_ = _e4035_;
                                            } else {
                                                let _e4190_ = select(0u, 2u, _e4040_);
                                                if _e4040_ {
                                                    phi_17173_ = (20u + _e4038_);
                                                } else {
                                                    phi_17173_ = (16u + _e4038_);
                                                }
                                                let _e4195_ = phi_17173_;
                                                let _e4196_ = (_e4195_ < 256u);
                                                if _e4196_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                let _e4198_ = aux_scratch[_e4195_];
                                                phi_17183_ = type_15_(0u, _e1581_);
                                                phi_17186_ = _e4198_;
                                                loop {
                                                    let _e4201_ = phi_17183_;
                                                    let _e4203_ = phi_17186_;
                                                    local_41_ = _e4203_;
                                                    if (_e4201_.member < _e4201_.member_1_) {
                                                        phi_17184_ = type_15_((_e4201_.member + 1u), _e4201_.member_1_);
                                                        phi_17209_ = type_15_(1u, _e4201_.member);
                                                    } else {
                                                        phi_17184_ = _e4201_;
                                                        phi_17209_ = type_15_(0u, u32());
                                                    }
                                                    let _e4216_ = phi_17184_;
                                                    let _e4218_ = phi_17209_;
                                                    let _e4222_ = (bitcast<i32>(_e4218_.member) != 0i);
                                                    if _e4222_ {
                                                        let _e4224_ = (_e4038_ * 4u);
                                                        let _e4227_ = ((_e4218_.member_1_ & 127u) & 127u);
                                                        let _e4228_ = (((_e4224_ + _e4190_) * 128u) + _e4227_);
                                                        if (_e4228_ < 2048u) {
                                                        } else {
                                                            phi_34428_ = true;
                                                            break;
                                                        }
                                                        let _e4231_ = lane_buf[_e4228_];
                                                        let _e4234_ = (((_e4224_ + (_e4190_ + 1u)) * 128u) + _e4227_);
                                                        let _e4235_ = (_e4234_ < 2048u);
                                                        if _e4235_ {
                                                        } else {
                                                            phi_34428_ = true;
                                                            break;
                                                        }
                                                        let _e4237_ = lane_buf[_e4234_];
                                                        let _e4240_ = bitcast<u32>((_e4178_ * _e4237_));
                                                        let _e4241_ = (_e4240_ & 2139095040u);
                                                        if (_e4241_ == 2139095040u) {
                                                            phi_17256_ = 0u;
                                                        } else {
                                                            if (_e4241_ == 0u) {
                                                                phi_17255_ = (_e4240_ & 2147483648u);
                                                            } else {
                                                                phi_17255_ = _e4240_;
                                                            }
                                                            let _e4246_ = phi_17255_;
                                                            phi_17256_ = _e4246_;
                                                        }
                                                        let _e4248_ = phi_17256_;
                                                        let _e4257_ = ((1f - _e4114_) + select((_e4152_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e4248_)) & 2147483647u))), 0f, (_e4152_ <= 0f)));
                                                        if (_e4257_ != _e4257_) {
                                                            phi_25100_ = true;
                                                        } else {
                                                            phi_25100_ = (0f >= _e4257_);
                                                        }
                                                        let _e4261_ = phi_25100_;
                                                        let _e4262_ = select(_e4257_, 0f, _e4261_);
                                                        if (_e4262_ != _e4262_) {
                                                            phi_25160_ = true;
                                                        } else {
                                                            phi_25160_ = (1f <= _e4262_);
                                                        }
                                                        let _e4266_ = phi_25160_;
                                                        let _e4267_ = select(_e4262_, 1f, _e4266_);
                                                        let _e4268_ = (_e4095_ * _e4203_);
                                                        let _e4269_ = (1f - _e4171_);
                                                        let _e4272_ = ((_e4185_ * _e4231_) - ((_e4133_ * _e4269_) * _e4268_));
                                                        if (_e4272_ != _e4272_) {
                                                            phi_25220_ = true;
                                                        } else {
                                                            phi_25220_ = (-1f >= _e4272_);
                                                        }
                                                        let _e4276_ = phi_25220_;
                                                        let _e4277_ = select(_e4272_, -1f, _e4276_);
                                                        if (_e4277_ != _e4277_) {
                                                            phi_25280_ = true;
                                                        } else {
                                                            phi_25280_ = (1f <= _e4277_);
                                                        }
                                                        let _e4281_ = phi_25280_;
                                                        let _e4282_ = select(_e4277_, 1f, _e4281_);
                                                        let _e4289_ = bitcast<f32>((bitcast<u32>(_e4272_) & 2147483647u));
                                                        if (_e4289_ != _e4289_) {
                                                            phi_25340_ = true;
                                                        } else {
                                                            phi_25340_ = (1000000000000000f <= _e4289_);
                                                        }
                                                        let _e4293_ = phi_25340_;
                                                        let _e4294_ = select(_e4289_, 1000000000000000f, _e4293_);
                                                        let _e4295_ = (_e4294_ * _e4294_);
                                                        let _e4296_ = (_e4295_ + 0.0000010000001f);
                                                        if (_e4296_ != _e4296_) {
                                                            phi_25400_ = true;
                                                        } else {
                                                            phi_25400_ = (0.000000000001f >= _e4296_);
                                                        }
                                                        let _e4300_ = phi_25400_;
                                                        let _e4301_ = select(_e4296_, 0.000000000001f, _e4300_);
                                                        let _e4302_ = (_e4301_ * 0.5f);
                                                        let _e4307_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e4301_) >> bitcast<u32>(1i))));
                                                        let _e4311_ = (_e4307_ * (1.5f - (_e4302_ * (_e4307_ * _e4307_))));
                                                        let _e4315_ = (_e4311_ * (1.5f - (_e4302_ * (_e4311_ * _e4311_))));
                                                        let _e4320_ = (_e4296_ * (_e4315_ * (1.5f - (_e4302_ * (_e4315_ * _e4315_)))));
                                                        let _e4321_ = (_e4320_ + 0.001f);
                                                        if (_e4321_ != _e4321_) {
                                                            phi_25460_ = true;
                                                        } else {
                                                            phi_25460_ = (0.000001f >= _e4321_);
                                                        }
                                                        let _e4325_ = phi_25460_;
                                                        let _e4326_ = select(_e4321_, 0.000001f, _e4325_);
                                                        let _e4329_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4326_)));
                                                        let _e4333_ = (_e4329_ + (_e4329_ * (1f - (_e4326_ * _e4329_))));
                                                        let _e4337_ = (_e4333_ + (_e4333_ * (1f - (_e4326_ * _e4333_))));
                                                        let _e4341_ = (_e4337_ + (_e4337_ * (1f - (_e4326_ * _e4337_))));
                                                        let _e4342_ = (_e4320_ + _e4294_);
                                                        if (_e4342_ != _e4342_) {
                                                            phi_25520_ = true;
                                                        } else {
                                                            phi_25520_ = (0.000001f >= _e4342_);
                                                        }
                                                        let _e4346_ = phi_25520_;
                                                        let _e4347_ = select(_e4342_, 0.000001f, _e4346_);
                                                        let _e4350_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4347_)));
                                                        let _e4354_ = (_e4350_ + (_e4350_ * (1f - (_e4347_ * _e4350_))));
                                                        let _e4358_ = (_e4354_ + (_e4354_ * (1f - (_e4347_ * _e4354_))));
                                                        let _e4368_ = (_e4289_ <= 1000000000000000f);
                                                        let _e4378_ = select(select(select(_e4289_, (0.5f * (_e4294_ + (_e4295_ * _e4341_))), _e4368_), select(-0.0005f, -(((0.0005f * (_e4294_ * (_e4358_ + (_e4358_ * (1f - (_e4347_ * _e4358_)))))) * (1f + (_e4294_ * _e4341_)))), _e4368_), (_e4272_ <= 0f)), 0f, (_e4289_ <= 0f));
                                                        let _e4389_ = bitcast<u32>((((1f - (_e4171_ * _e4267_)) * _e4268_) + (_e4267_ * select(((_e4269_ * _e4378_) + (_e4171_ * (_e4282_ * (1f - (0.33333334f * (_e4282_ * _e4282_)))))), _e4378_, (_e4171_ <= 0f)))));
                                                        let _e4390_ = (_e4389_ & 2139095040u);
                                                        if (_e4390_ == 2139095040u) {
                                                            phi_17998_ = 0u;
                                                        } else {
                                                            if (_e4390_ == 0u) {
                                                                phi_17997_ = (_e4389_ & 2147483648u);
                                                            } else {
                                                                phi_17997_ = _e4389_;
                                                            }
                                                            let _e4395_ = phi_17997_;
                                                            phi_17998_ = _e4395_;
                                                        }
                                                        let _e4397_ = phi_17998_;
                                                        let _e4398_ = bitcast<f32>(_e4397_);
                                                        if _e4235_ {
                                                        } else {
                                                            phi_34428_ = true;
                                                            break;
                                                        }
                                                        lane_buf[_e4234_] = _e4398_;
                                                        phi_17187_ = _e4398_;
                                                    } else {
                                                        phi_17187_ = f32();
                                                    }
                                                    let _e4400_ = phi_17187_;
                                                    phi_17183_ = _e4216_;
                                                    phi_17186_ = _e4400_;
                                                    phi_34428_ = _e4035_;
                                                    if !(_e4222_) {
                                                        break;
                                                    }
                                                }
                                                let _e4403_ = phi_34428_;
                                                phi_36246_ = _e4403_;
                                                if _e4403_ {
                                                    break;
                                                }
                                                let _e4405_ = local_41_;
                                                let _e4406_ = bitcast<u32>(_e4405_);
                                                let _e4407_ = (_e4406_ & 2139095040u);
                                                if (_e4407_ == 2139095040u) {
                                                    phi_18052_ = true;
                                                } else {
                                                    phi_18052_ = (_e4407_ == 0u);
                                                }
                                                let _e4411_ = phi_18052_;
                                                if _e4196_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                aux_scratch[_e4195_] = bitcast<f32>(select(_e4406_, 0u, _e4411_));
                                                phi_35384_ = _e4403_;
                                            }
                                            let _e4415_ = phi_35384_;
                                            phi_35383_ = _e4415_;
                                        } else {
                                            phi_35383_ = _e4035_;
                                        }
                                        let _e4417_ = phi_35383_;
                                        workgroupBarrier();
                                        phi_35382_ = _e4417_;
                                    }
                                    let _e6403_ = phi_35382_;
                                    local_9_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    local_10_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    local_11_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    local_12_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    let _e6404_ = (_e163_ < 128u);
                                    if _e6404_ {
                                        if _e1583_ {
                                            local_13_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_14_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_15_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_16_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            phi_18073_ = type_15_(0u, 4u);
                                            loop {
                                                let _e6406_ = phi_18073_;
                                                if (_e6406_.member < _e6406_.member_1_) {
                                                    phi_18074_ = type_15_((_e6406_.member + 1u), _e6406_.member_1_);
                                                    phi_18097_ = type_15_(1u, _e6406_.member);
                                                } else {
                                                    phi_18074_ = _e6406_;
                                                    phi_18097_ = type_15_(0u, u32());
                                                }
                                                let _e6419_ = phi_18074_;
                                                let _e6421_ = phi_18097_;
                                                let _e6425_ = (bitcast<i32>(_e6421_.member) != 0i);
                                                if _e6425_ {
                                                    let _e6426_ = (_e6421_.member_1_ * 4u);
                                                    let _e6428_ = (_e164_ & 127u);
                                                    let _e6429_ = ((_e6421_.member_1_ * 512u) + _e6428_);
                                                    if (_e6429_ < 2048u) {
                                                    } else {
                                                        phi_35378_ = true;
                                                        break;
                                                    }
                                                    let _e6432_ = lane_buf[_e6429_];
                                                    let _e6433_ = (_e6421_.member_1_ < 4u);
                                                    if _e6433_ {
                                                    } else {
                                                        phi_35378_ = true;
                                                        break;
                                                    }
                                                    local_13_[_e6421_.member_1_] = _e6432_;
                                                    let _e6437_ = (((_e6426_ + 1u) * 128u) + _e6428_);
                                                    if (_e6437_ < 2048u) {
                                                    } else {
                                                        phi_35378_ = true;
                                                        break;
                                                    }
                                                    let _e6440_ = lane_buf[_e6437_];
                                                    if _e6433_ {
                                                    } else {
                                                        phi_35378_ = true;
                                                        break;
                                                    }
                                                    local_15_[_e6421_.member_1_] = _e6440_;
                                                    let _e6444_ = (((_e6426_ + 2u) * 128u) + _e6428_);
                                                    if (_e6444_ < 2048u) {
                                                    } else {
                                                        phi_35378_ = true;
                                                        break;
                                                    }
                                                    let _e6447_ = lane_buf[_e6444_];
                                                    if _e6433_ {
                                                    } else {
                                                        phi_35378_ = true;
                                                        break;
                                                    }
                                                    local_14_[_e6421_.member_1_] = _e6447_;
                                                    let _e6451_ = (((_e6426_ + 3u) * 128u) + _e6428_);
                                                    if (_e6451_ < 2048u) {
                                                    } else {
                                                        phi_35378_ = true;
                                                        break;
                                                    }
                                                    let _e6454_ = lane_buf[_e6451_];
                                                    if _e6433_ {
                                                    } else {
                                                        phi_35378_ = true;
                                                        break;
                                                    }
                                                    local_16_[_e6421_.member_1_] = _e6454_;
                                                }
                                                phi_18073_ = _e6419_;
                                                phi_35378_ = _e6403_;
                                                if !(_e6425_) {
                                                    break;
                                                }
                                            }
                                            let _e6458_ = phi_35378_;
                                            phi_36246_ = _e6458_;
                                            if _e6458_ {
                                                break;
                                            }
                                            let _e6459_ = local_15_;
                                            let _e6464_ = local_16_;
                                            let _e6473_ = (0.5f * (_e6459_[0] + _e6464_[0]));
                                            let _e6474_ = (0.5f * (_e6459_[1] + _e6464_[1]));
                                            let _e6475_ = (0.5f * (_e6459_[2] + _e6464_[2]));
                                            let _e6476_ = (0.5f * (_e6459_[3] + _e6464_[3]));
                                            local_17_ = array<f32, 4>((_e6459_[0] + (_e569_ * (_e6473_ - _e6459_[0]))), (_e6459_[1] + (_e589_ * (_e6474_ - _e6459_[1]))), (_e6459_[2] + (_e609_ * (_e6475_ - _e6459_[2]))), (_e6459_[3] + (_e629_ * (_e6476_ - _e6459_[3]))));
                                            local_18_ = array<f32, 4>((_e6464_[0] + (_e569_ * (_e6473_ - _e6464_[0]))), (_e6464_[1] + (_e589_ * (_e6474_ - _e6464_[1]))), (_e6464_[2] + (_e609_ * (_e6475_ - _e6464_[2]))), (_e6464_[3] + (_e629_ * (_e6476_ - _e6464_[3]))));
                                            phi_18269_ = type_15_(0u, 4u);
                                            loop {
                                                let _e6504_ = phi_18269_;
                                                if (_e6504_.member < _e6504_.member_1_) {
                                                    phi_18270_ = type_15_((_e6504_.member + 1u), _e6504_.member_1_);
                                                    phi_18293_ = type_15_(1u, _e6504_.member);
                                                } else {
                                                    phi_18270_ = _e6504_;
                                                    phi_18293_ = type_15_(0u, u32());
                                                }
                                                let _e6517_ = phi_18270_;
                                                let _e6519_ = phi_18293_;
                                                let _e6523_ = (bitcast<i32>(_e6519_.member) != 0i);
                                                if _e6523_ {
                                                    let _e6524_ = (_e6519_.member_1_ & 3u);
                                                    switch bitcast<i32>(_e6524_) {
                                                        case 0: {
                                                            phi_18664_ = _e432_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_18664_ = _e432_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_18664_ = _e432_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_18664_ = _e432_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e6531_ = phi_18664_;
                                                    switch bitcast<i32>(_e6524_) {
                                                        case 0: {
                                                            phi_18675_ = _e439_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_18675_ = _e439_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_18675_ = _e439_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_18675_ = _e439_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e6538_ = phi_18675_;
                                                    switch bitcast<i32>(_e6524_) {
                                                        case 0: {
                                                            phi_18686_ = _e446_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_18686_ = _e446_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_18686_ = _e446_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_18686_ = _e446_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e6545_ = phi_18686_;
                                                    let _e6546_ = (_e6519_.member_1_ < 4u);
                                                    if _e6546_ {
                                                    } else {
                                                        phi_35387_ = true;
                                                        break;
                                                    }
                                                    let _e6548_ = local_13_[_e6519_.member_1_];
                                                    if _e6546_ {
                                                    } else {
                                                        phi_35387_ = true;
                                                        break;
                                                    }
                                                    let _e6550_ = local_14_[_e6519_.member_1_];
                                                    if _e6546_ {
                                                    } else {
                                                        phi_35387_ = true;
                                                        break;
                                                    }
                                                    let _e6552_ = local_17_[_e6519_.member_1_];
                                                    if _e6546_ {
                                                    } else {
                                                        phi_35387_ = true;
                                                        break;
                                                    }
                                                    let _e6554_ = local_18_[_e6519_.member_1_];
                                                    if _e6546_ {
                                                    } else {
                                                        phi_35387_ = true;
                                                        break;
                                                    }
                                                    local_9_[_e6519_.member_1_] = _e6552_;
                                                    if _e6546_ {
                                                    } else {
                                                        phi_35387_ = true;
                                                        break;
                                                    }
                                                    local_10_[_e6519_.member_1_] = _e6554_;
                                                    let _e6558_ = (1f - (_e6531_ * _e6552_));
                                                    if (_e6558_ != _e6558_) {
                                                        phi_29540_ = true;
                                                    } else {
                                                        phi_29540_ = (0.05f >= _e6558_);
                                                    }
                                                    let _e6562_ = phi_29540_;
                                                    let _e6565_ = (1f - (_e6531_ * _e6554_));
                                                    if (_e6565_ != _e6565_) {
                                                        phi_29555_ = true;
                                                    } else {
                                                        phi_29555_ = (0.05f >= _e6565_);
                                                    }
                                                    let _e6569_ = phi_29555_;
                                                    let _e6575_ = bitcast<u32>((((_e6548_ * _e6538_) * select(_e6558_, 0.05f, _e6562_)) + (_e6545_ * _e6552_)));
                                                    let _e6576_ = (_e6575_ & 2139095040u);
                                                    if (_e6576_ == 2139095040u) {
                                                        phi_18793_ = true;
                                                    } else {
                                                        phi_18793_ = (_e6576_ == 0u);
                                                    }
                                                    let _e6580_ = phi_18793_;
                                                    let _e6582_ = bitcast<f32>(select(_e6575_, 0u, _e6580_));
                                                    let _e6587_ = bitcast<u32>((((_e6550_ * _e6538_) * select(_e6565_, 0.05f, _e6569_)) + (_e6545_ * _e6554_)));
                                                    let _e6588_ = (_e6587_ & 2139095040u);
                                                    if (_e6588_ == 2139095040u) {
                                                        phi_18813_ = true;
                                                    } else {
                                                        phi_18813_ = (_e6588_ == 0u);
                                                    }
                                                    let _e6592_ = phi_18813_;
                                                    let _e6594_ = bitcast<f32>(select(_e6587_, 0u, _e6592_));
                                                    if _e6546_ {
                                                    } else {
                                                        phi_35387_ = true;
                                                        break;
                                                    }
                                                    local_11_[_e6519_.member_1_] = _e6582_;
                                                    if _e6546_ {
                                                    } else {
                                                        phi_35387_ = true;
                                                        break;
                                                    }
                                                    local_12_[_e6519_.member_1_] = _e6594_;
                                                    let _e6598_ = ((_e6519_.member_1_ * 128u) + _e164_);
                                                    if (_e6598_ < 1536u) {
                                                    } else {
                                                        phi_35387_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e6598_] = _e6582_;
                                                    let _e6603_ = (((4u + _e6519_.member_1_) * 128u) + _e164_);
                                                    if (_e6603_ < 1536u) {
                                                    } else {
                                                        phi_35387_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e6603_] = _e6594_;
                                                }
                                                phi_18269_ = _e6517_;
                                                phi_35387_ = _e6458_;
                                                if !(_e6523_) {
                                                    break;
                                                }
                                            }
                                            let _e6608_ = phi_35387_;
                                            phi_36246_ = _e6608_;
                                            if _e6608_ {
                                                break;
                                            }
                                            phi_35407_ = _e6608_;
                                        } else {
                                            phi_35407_ = _e6403_;
                                        }
                                        let _e6610_ = phi_35407_;
                                        phi_35406_ = _e6610_;
                                    } else {
                                        phi_35406_ = _e6403_;
                                    }
                                    let _e6612_ = phi_35406_;
                                    workgroupBarrier();
                                    if _e6404_ {
                                        if _e1583_ {
                                            local_19_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_20_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            if (_e164_ == 0u) {
                                                phi_18861_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e6654_ = phi_18861_;
                                                    if (_e6654_.member < _e6654_.member_1_) {
                                                        phi_18862_ = type_15_((_e6654_.member + 1u), _e6654_.member_1_);
                                                        phi_18885_ = type_15_(1u, _e6654_.member);
                                                    } else {
                                                        phi_18862_ = _e6654_;
                                                        phi_18885_ = type_15_(0u, u32());
                                                    }
                                                    let _e6667_ = phi_18862_;
                                                    let _e6669_ = phi_18885_;
                                                    let _e6673_ = (bitcast<i32>(_e6669_.member) != 0i);
                                                    if _e6673_ {
                                                        let _e6674_ = (24u + _e6669_.member_1_);
                                                        if (_e6674_ < 256u) {
                                                        } else {
                                                            phi_35408_ = true;
                                                            break;
                                                        }
                                                        let _e6677_ = aux_scratch[_e6674_];
                                                        let _e6678_ = (_e6669_.member_1_ < 4u);
                                                        if _e6678_ {
                                                        } else {
                                                            phi_35408_ = true;
                                                            break;
                                                        }
                                                        local_19_[_e6669_.member_1_] = _e6677_;
                                                        let _e6680_ = (28u + _e6669_.member_1_);
                                                        if (_e6680_ < 256u) {
                                                        } else {
                                                            phi_35408_ = true;
                                                            break;
                                                        }
                                                        let _e6683_ = aux_scratch[_e6680_];
                                                        if _e6678_ {
                                                        } else {
                                                            phi_35408_ = true;
                                                            break;
                                                        }
                                                        local_20_[_e6669_.member_1_] = _e6683_;
                                                    }
                                                    phi_18861_ = _e6667_;
                                                    phi_35408_ = _e6612_;
                                                    if !(_e6673_) {
                                                        break;
                                                    }
                                                }
                                                let _e6687_ = phi_35408_;
                                                phi_36246_ = _e6687_;
                                                if _e6687_ {
                                                    break;
                                                }
                                                phi_36231_ = _e6687_;
                                            } else {
                                                let _e6614_ = (_e164_ - 1u);
                                                phi_18924_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e6616_ = phi_18924_;
                                                    if (_e6616_.member < _e6616_.member_1_) {
                                                        phi_18925_ = type_15_((_e6616_.member + 1u), _e6616_.member_1_);
                                                        phi_18948_ = type_15_(1u, _e6616_.member);
                                                    } else {
                                                        phi_18925_ = _e6616_;
                                                        phi_18948_ = type_15_(0u, u32());
                                                    }
                                                    let _e6629_ = phi_18925_;
                                                    let _e6631_ = phi_18948_;
                                                    let _e6635_ = (bitcast<i32>(_e6631_.member) != 0i);
                                                    if _e6635_ {
                                                        let _e6637_ = ((_e6631_.member_1_ * 128u) + _e6614_);
                                                        if (_e6637_ < 1536u) {
                                                        } else {
                                                            phi_35402_ = true;
                                                            break;
                                                        }
                                                        let _e6640_ = scan_scratch[_e6637_];
                                                        let _e6641_ = (_e6631_.member_1_ < 4u);
                                                        if _e6641_ {
                                                        } else {
                                                            phi_35402_ = true;
                                                            break;
                                                        }
                                                        local_19_[_e6631_.member_1_] = _e6640_;
                                                        let _e6645_ = (((4u + _e6631_.member_1_) * 128u) + _e6614_);
                                                        if (_e6645_ < 1536u) {
                                                        } else {
                                                            phi_35402_ = true;
                                                            break;
                                                        }
                                                        let _e6648_ = scan_scratch[_e6645_];
                                                        if _e6641_ {
                                                        } else {
                                                            phi_35402_ = true;
                                                            break;
                                                        }
                                                        local_20_[_e6631_.member_1_] = _e6648_;
                                                    }
                                                    phi_18924_ = _e6629_;
                                                    phi_35402_ = _e6612_;
                                                    if !(_e6635_) {
                                                        break;
                                                    }
                                                }
                                                let _e6652_ = phi_35402_;
                                                phi_36246_ = _e6652_;
                                                if _e6652_ {
                                                    break;
                                                }
                                                phi_36231_ = _e6652_;
                                            }
                                            let _e6689_ = phi_36231_;
                                            let _e6690_ = local_19_;
                                            let _e6695_ = local_11_;
                                            let _e6700_ = local_20_;
                                            let _e6705_ = local_12_;
                                            let _e6723_ = select(0f, _e6690_[0], (bitcast<f32>((bitcast<u32>(_e6690_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6725_ = select(0f, _e6690_[1], (bitcast<f32>((bitcast<u32>(_e6690_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6727_ = select(0f, _e6690_[2], (bitcast<f32>((bitcast<u32>(_e6690_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6729_ = select(0f, _e6690_[3], (bitcast<f32>((bitcast<u32>(_e6690_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6730_ = (_e6723_ != _e6723_);
                                            if _e6730_ {
                                                phi_29570_ = true;
                                            } else {
                                                phi_29570_ = (-1000000000000000000f >= _e6723_);
                                            }
                                            let _e6733_ = phi_29570_;
                                            let _e6734_ = select(_e6723_, -1000000000000000000f, _e6733_);
                                            let _e6735_ = (_e6725_ != _e6725_);
                                            if _e6735_ {
                                                phi_29585_ = true;
                                            } else {
                                                phi_29585_ = (-1000000000000000000f >= _e6725_);
                                            }
                                            let _e6738_ = phi_29585_;
                                            let _e6739_ = select(_e6725_, -1000000000000000000f, _e6738_);
                                            let _e6740_ = (_e6727_ != _e6727_);
                                            if _e6740_ {
                                                phi_29600_ = true;
                                            } else {
                                                phi_29600_ = (-1000000000000000000f >= _e6727_);
                                            }
                                            let _e6743_ = phi_29600_;
                                            let _e6744_ = select(_e6727_, -1000000000000000000f, _e6743_);
                                            let _e6745_ = (_e6729_ != _e6729_);
                                            if _e6745_ {
                                                phi_29615_ = true;
                                            } else {
                                                phi_29615_ = (-1000000000000000000f >= _e6729_);
                                            }
                                            let _e6748_ = phi_29615_;
                                            let _e6749_ = select(_e6729_, -1000000000000000000f, _e6748_);
                                            if (_e6734_ != _e6734_) {
                                                phi_29630_ = true;
                                            } else {
                                                phi_29630_ = (1000000000000000000f <= _e6734_);
                                            }
                                            let _e6753_ = phi_29630_;
                                            let _e6754_ = select(_e6734_, 1000000000000000000f, _e6753_);
                                            if (_e6739_ != _e6739_) {
                                                phi_29645_ = true;
                                            } else {
                                                phi_29645_ = (1000000000000000000f <= _e6739_);
                                            }
                                            let _e6758_ = phi_29645_;
                                            let _e6759_ = select(_e6739_, 1000000000000000000f, _e6758_);
                                            if (_e6744_ != _e6744_) {
                                                phi_29660_ = true;
                                            } else {
                                                phi_29660_ = (1000000000000000000f <= _e6744_);
                                            }
                                            let _e6763_ = phi_29660_;
                                            let _e6764_ = select(_e6744_, 1000000000000000000f, _e6763_);
                                            if (_e6749_ != _e6749_) {
                                                phi_29675_ = true;
                                            } else {
                                                phi_29675_ = (1000000000000000000f <= _e6749_);
                                            }
                                            let _e6768_ = phi_29675_;
                                            let _e6769_ = select(_e6749_, 1000000000000000000f, _e6768_);
                                            let _e6783_ = select(0f, _e6695_[0], (bitcast<f32>((bitcast<u32>(_e6695_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6785_ = select(0f, _e6695_[1], (bitcast<f32>((bitcast<u32>(_e6695_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6787_ = select(0f, _e6695_[2], (bitcast<f32>((bitcast<u32>(_e6695_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6789_ = select(0f, _e6695_[3], (bitcast<f32>((bitcast<u32>(_e6695_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6790_ = (_e6783_ != _e6783_);
                                            if _e6790_ {
                                                phi_29690_ = true;
                                            } else {
                                                phi_29690_ = (-1000000000000000000f >= _e6783_);
                                            }
                                            let _e6793_ = phi_29690_;
                                            let _e6794_ = select(_e6783_, -1000000000000000000f, _e6793_);
                                            let _e6795_ = (_e6785_ != _e6785_);
                                            if _e6795_ {
                                                phi_29705_ = true;
                                            } else {
                                                phi_29705_ = (-1000000000000000000f >= _e6785_);
                                            }
                                            let _e6798_ = phi_29705_;
                                            let _e6799_ = select(_e6785_, -1000000000000000000f, _e6798_);
                                            let _e6800_ = (_e6787_ != _e6787_);
                                            if _e6800_ {
                                                phi_29720_ = true;
                                            } else {
                                                phi_29720_ = (-1000000000000000000f >= _e6787_);
                                            }
                                            let _e6803_ = phi_29720_;
                                            let _e6804_ = select(_e6787_, -1000000000000000000f, _e6803_);
                                            let _e6805_ = (_e6789_ != _e6789_);
                                            if _e6805_ {
                                                phi_29735_ = true;
                                            } else {
                                                phi_29735_ = (-1000000000000000000f >= _e6789_);
                                            }
                                            let _e6808_ = phi_29735_;
                                            let _e6809_ = select(_e6789_, -1000000000000000000f, _e6808_);
                                            if (_e6794_ != _e6794_) {
                                                phi_29750_ = true;
                                            } else {
                                                phi_29750_ = (1000000000000000000f <= _e6794_);
                                            }
                                            let _e6813_ = phi_29750_;
                                            let _e6814_ = select(_e6794_, 1000000000000000000f, _e6813_);
                                            if (_e6799_ != _e6799_) {
                                                phi_29765_ = true;
                                            } else {
                                                phi_29765_ = (1000000000000000000f <= _e6799_);
                                            }
                                            let _e6818_ = phi_29765_;
                                            let _e6819_ = select(_e6799_, 1000000000000000000f, _e6818_);
                                            if (_e6804_ != _e6804_) {
                                                phi_29780_ = true;
                                            } else {
                                                phi_29780_ = (1000000000000000000f <= _e6804_);
                                            }
                                            let _e6823_ = phi_29780_;
                                            let _e6824_ = select(_e6804_, 1000000000000000000f, _e6823_);
                                            if (_e6809_ != _e6809_) {
                                                phi_29795_ = true;
                                            } else {
                                                phi_29795_ = (1000000000000000000f <= _e6809_);
                                            }
                                            let _e6828_ = phi_29795_;
                                            let _e6829_ = select(_e6809_, 1000000000000000000f, _e6828_);
                                            let _e6830_ = array<f32, 4>(_e6814_, _e6819_, _e6824_, _e6829_);
                                            let _e6831_ = (_e6754_ != _e6754_);
                                            if _e6831_ {
                                                phi_29810_ = true;
                                            } else {
                                                phi_29810_ = (_e6814_ <= _e6754_);
                                            }
                                            let _e6834_ = phi_29810_;
                                            let _e6835_ = select(_e6754_, _e6814_, _e6834_);
                                            let _e6836_ = (_e6759_ != _e6759_);
                                            if _e6836_ {
                                                phi_29825_ = true;
                                            } else {
                                                phi_29825_ = (_e6819_ <= _e6759_);
                                            }
                                            let _e6839_ = phi_29825_;
                                            let _e6840_ = select(_e6759_, _e6819_, _e6839_);
                                            let _e6841_ = (_e6764_ != _e6764_);
                                            if _e6841_ {
                                                phi_29840_ = true;
                                            } else {
                                                phi_29840_ = (_e6824_ <= _e6764_);
                                            }
                                            let _e6844_ = phi_29840_;
                                            let _e6845_ = select(_e6764_, _e6824_, _e6844_);
                                            let _e6846_ = (_e6769_ != _e6769_);
                                            if _e6846_ {
                                                phi_29855_ = true;
                                            } else {
                                                phi_29855_ = (_e6829_ <= _e6769_);
                                            }
                                            let _e6849_ = phi_29855_;
                                            let _e6850_ = select(_e6769_, _e6829_, _e6849_);
                                            if _e6831_ {
                                                phi_29870_ = true;
                                            } else {
                                                phi_29870_ = (_e6814_ >= _e6754_);
                                            }
                                            let _e6853_ = phi_29870_;
                                            let _e6854_ = select(_e6754_, _e6814_, _e6853_);
                                            if _e6836_ {
                                                phi_29885_ = true;
                                            } else {
                                                phi_29885_ = (_e6819_ >= _e6759_);
                                            }
                                            let _e6857_ = phi_29885_;
                                            let _e6858_ = select(_e6759_, _e6819_, _e6857_);
                                            if _e6841_ {
                                                phi_29900_ = true;
                                            } else {
                                                phi_29900_ = (_e6824_ >= _e6764_);
                                            }
                                            let _e6861_ = phi_29900_;
                                            let _e6862_ = select(_e6764_, _e6824_, _e6861_);
                                            if _e6846_ {
                                                phi_29915_ = true;
                                            } else {
                                                phi_29915_ = (_e6829_ >= _e6769_);
                                            }
                                            let _e6865_ = phi_29915_;
                                            let _e6866_ = select(_e6769_, _e6829_, _e6865_);
                                            let _e6867_ = (_e6835_ != _e6835_);
                                            if _e6867_ {
                                                phi_29930_ = true;
                                            } else {
                                                phi_29930_ = (-1f >= _e6835_);
                                            }
                                            let _e6870_ = phi_29930_;
                                            let _e6871_ = select(_e6835_, -1f, _e6870_);
                                            let _e6872_ = (_e6840_ != _e6840_);
                                            if _e6872_ {
                                                phi_29945_ = true;
                                            } else {
                                                phi_29945_ = (-1f >= _e6840_);
                                            }
                                            let _e6875_ = phi_29945_;
                                            let _e6876_ = select(_e6840_, -1f, _e6875_);
                                            let _e6877_ = (_e6845_ != _e6845_);
                                            if _e6877_ {
                                                phi_29960_ = true;
                                            } else {
                                                phi_29960_ = (-1f >= _e6845_);
                                            }
                                            let _e6880_ = phi_29960_;
                                            let _e6881_ = select(_e6845_, -1f, _e6880_);
                                            let _e6882_ = (_e6850_ != _e6850_);
                                            if _e6882_ {
                                                phi_29975_ = true;
                                            } else {
                                                phi_29975_ = (-1f >= _e6850_);
                                            }
                                            let _e6885_ = phi_29975_;
                                            let _e6886_ = select(_e6850_, -1f, _e6885_);
                                            if (_e6871_ != _e6871_) {
                                                phi_29990_ = true;
                                            } else {
                                                phi_29990_ = (1f <= _e6871_);
                                            }
                                            let _e6890_ = phi_29990_;
                                            let _e6891_ = select(_e6871_, 1f, _e6890_);
                                            if (_e6876_ != _e6876_) {
                                                phi_30005_ = true;
                                            } else {
                                                phi_30005_ = (1f <= _e6876_);
                                            }
                                            let _e6895_ = phi_30005_;
                                            let _e6896_ = select(_e6876_, 1f, _e6895_);
                                            if (_e6881_ != _e6881_) {
                                                phi_30020_ = true;
                                            } else {
                                                phi_30020_ = (1f <= _e6881_);
                                            }
                                            let _e6900_ = phi_30020_;
                                            let _e6901_ = select(_e6881_, 1f, _e6900_);
                                            if (_e6886_ != _e6886_) {
                                                phi_30035_ = true;
                                            } else {
                                                phi_30035_ = (1f <= _e6886_);
                                            }
                                            let _e6905_ = phi_30035_;
                                            let _e6906_ = select(_e6886_, 1f, _e6905_);
                                            let _e6907_ = (_e6854_ != _e6854_);
                                            if _e6907_ {
                                                phi_30050_ = true;
                                            } else {
                                                phi_30050_ = (-1f >= _e6854_);
                                            }
                                            let _e6910_ = phi_30050_;
                                            let _e6911_ = select(_e6854_, -1f, _e6910_);
                                            let _e6912_ = (_e6858_ != _e6858_);
                                            if _e6912_ {
                                                phi_30065_ = true;
                                            } else {
                                                phi_30065_ = (-1f >= _e6858_);
                                            }
                                            let _e6915_ = phi_30065_;
                                            let _e6916_ = select(_e6858_, -1f, _e6915_);
                                            let _e6917_ = (_e6862_ != _e6862_);
                                            if _e6917_ {
                                                phi_30080_ = true;
                                            } else {
                                                phi_30080_ = (-1f >= _e6862_);
                                            }
                                            let _e6920_ = phi_30080_;
                                            let _e6921_ = select(_e6862_, -1f, _e6920_);
                                            let _e6922_ = (_e6866_ != _e6866_);
                                            if _e6922_ {
                                                phi_30095_ = true;
                                            } else {
                                                phi_30095_ = (-1f >= _e6866_);
                                            }
                                            let _e6925_ = phi_30095_;
                                            let _e6926_ = select(_e6866_, -1f, _e6925_);
                                            if (_e6911_ != _e6911_) {
                                                phi_30110_ = true;
                                            } else {
                                                phi_30110_ = (1f <= _e6911_);
                                            }
                                            let _e6930_ = phi_30110_;
                                            let _e6931_ = select(_e6911_, 1f, _e6930_);
                                            if (_e6916_ != _e6916_) {
                                                phi_30125_ = true;
                                            } else {
                                                phi_30125_ = (1f <= _e6916_);
                                            }
                                            let _e6935_ = phi_30125_;
                                            let _e6936_ = select(_e6916_, 1f, _e6935_);
                                            if (_e6921_ != _e6921_) {
                                                phi_30140_ = true;
                                            } else {
                                                phi_30140_ = (1f <= _e6921_);
                                            }
                                            let _e6940_ = phi_30140_;
                                            let _e6941_ = select(_e6921_, 1f, _e6940_);
                                            if (_e6926_ != _e6926_) {
                                                phi_30155_ = true;
                                            } else {
                                                phi_30155_ = (1f <= _e6926_);
                                            }
                                            let _e6945_ = phi_30155_;
                                            let _e6946_ = select(_e6926_, 1f, _e6945_);
                                            let _e6951_ = (0.5f * (_e6931_ + _e6891_));
                                            let _e6952_ = (0.5f * (_e6936_ + _e6896_));
                                            let _e6953_ = (0.5f * (_e6941_ + _e6901_));
                                            let _e6954_ = (0.5f * (_e6946_ + _e6906_));
                                            let _e6955_ = (_e6931_ - _e6891_);
                                            let _e6956_ = (_e6936_ - _e6896_);
                                            let _e6957_ = (_e6941_ - _e6901_);
                                            let _e6958_ = (_e6946_ - _e6906_);
                                            let _e6959_ = (0.5f * _e6955_);
                                            let _e6960_ = (0.5f * _e6956_);
                                            let _e6961_ = (0.5f * _e6957_);
                                            let _e6962_ = (0.5f * _e6958_);
                                            let _e6963_ = (_e6951_ * _e6951_);
                                            let _e6964_ = (_e6952_ * _e6952_);
                                            let _e6965_ = (_e6953_ * _e6953_);
                                            let _e6966_ = (_e6954_ * _e6954_);
                                            let _e6967_ = (_e6959_ * _e6959_);
                                            let _e6968_ = (_e6960_ * _e6960_);
                                            let _e6969_ = (_e6961_ * _e6961_);
                                            let _e6970_ = (_e6962_ * _e6962_);
                                            if (_e6955_ != _e6955_) {
                                                phi_30170_ = true;
                                            } else {
                                                phi_30170_ = (0f >= _e6955_);
                                            }
                                            let _e6974_ = phi_30170_;
                                            let _e6975_ = select(_e6955_, 0f, _e6974_);
                                            if (_e6956_ != _e6956_) {
                                                phi_30185_ = true;
                                            } else {
                                                phi_30185_ = (0f >= _e6956_);
                                            }
                                            let _e6979_ = phi_30185_;
                                            let _e6980_ = select(_e6956_, 0f, _e6979_);
                                            if (_e6957_ != _e6957_) {
                                                phi_30200_ = true;
                                            } else {
                                                phi_30200_ = (0f >= _e6957_);
                                            }
                                            let _e6984_ = phi_30200_;
                                            let _e6985_ = select(_e6957_, 0f, _e6984_);
                                            if (_e6958_ != _e6958_) {
                                                phi_30215_ = true;
                                            } else {
                                                phi_30215_ = (0f >= _e6958_);
                                            }
                                            let _e6989_ = phi_30215_;
                                            let _e6990_ = select(_e6958_, 0f, _e6989_);
                                            let _e6991_ = (_e6854_ - _e6835_);
                                            let _e6992_ = (_e6858_ - _e6840_);
                                            let _e6993_ = (_e6862_ - _e6845_);
                                            let _e6994_ = (_e6866_ - _e6850_);
                                            if _e6907_ {
                                                phi_30230_ = true;
                                            } else {
                                                phi_30230_ = (-1f <= _e6854_);
                                            }
                                            let _e6997_ = phi_30230_;
                                            if _e6912_ {
                                                phi_30245_ = true;
                                            } else {
                                                phi_30245_ = (-1f <= _e6858_);
                                            }
                                            let _e7001_ = phi_30245_;
                                            if _e6917_ {
                                                phi_30260_ = true;
                                            } else {
                                                phi_30260_ = (-1f <= _e6862_);
                                            }
                                            let _e7005_ = phi_30260_;
                                            if _e6922_ {
                                                phi_30275_ = true;
                                            } else {
                                                phi_30275_ = (-1f <= _e6866_);
                                            }
                                            let _e7009_ = phi_30275_;
                                            let _e7011_ = (select(_e6854_, -1f, _e6997_) - _e6835_);
                                            let _e7012_ = (select(_e6858_, -1f, _e7001_) - _e6840_);
                                            let _e7013_ = (select(_e6862_, -1f, _e7005_) - _e6845_);
                                            let _e7014_ = (select(_e6866_, -1f, _e7009_) - _e6850_);
                                            if (_e7011_ != _e7011_) {
                                                phi_30290_ = true;
                                            } else {
                                                phi_30290_ = (0f >= _e7011_);
                                            }
                                            let _e7018_ = phi_30290_;
                                            if (_e7012_ != _e7012_) {
                                                phi_30305_ = true;
                                            } else {
                                                phi_30305_ = (0f >= _e7012_);
                                            }
                                            let _e7023_ = phi_30305_;
                                            if (_e7013_ != _e7013_) {
                                                phi_30320_ = true;
                                            } else {
                                                phi_30320_ = (0f >= _e7013_);
                                            }
                                            let _e7028_ = phi_30320_;
                                            if (_e7014_ != _e7014_) {
                                                phi_30335_ = true;
                                            } else {
                                                phi_30335_ = (0f >= _e7014_);
                                            }
                                            let _e7033_ = phi_30335_;
                                            if _e6867_ {
                                                phi_30350_ = true;
                                            } else {
                                                phi_30350_ = (1f >= _e6835_);
                                            }
                                            let _e7037_ = phi_30350_;
                                            if _e6872_ {
                                                phi_30365_ = true;
                                            } else {
                                                phi_30365_ = (1f >= _e6840_);
                                            }
                                            let _e7041_ = phi_30365_;
                                            if _e6877_ {
                                                phi_30380_ = true;
                                            } else {
                                                phi_30380_ = (1f >= _e6845_);
                                            }
                                            let _e7045_ = phi_30380_;
                                            if _e6882_ {
                                                phi_30395_ = true;
                                            } else {
                                                phi_30395_ = (1f >= _e6850_);
                                            }
                                            let _e7049_ = phi_30395_;
                                            let _e7051_ = (_e6854_ - select(_e6835_, 1f, _e7037_));
                                            let _e7052_ = (_e6858_ - select(_e6840_, 1f, _e7041_));
                                            let _e7053_ = (_e6862_ - select(_e6845_, 1f, _e7045_));
                                            let _e7054_ = (_e6866_ - select(_e6850_, 1f, _e7049_));
                                            if (_e7051_ != _e7051_) {
                                                phi_30410_ = true;
                                            } else {
                                                phi_30410_ = (0f >= _e7051_);
                                            }
                                            let _e7058_ = phi_30410_;
                                            if (_e7052_ != _e7052_) {
                                                phi_30425_ = true;
                                            } else {
                                                phi_30425_ = (0f >= _e7052_);
                                            }
                                            let _e7063_ = phi_30425_;
                                            if (_e7053_ != _e7053_) {
                                                phi_30440_ = true;
                                            } else {
                                                phi_30440_ = (0f >= _e7053_);
                                            }
                                            let _e7068_ = phi_30440_;
                                            if (_e7054_ != _e7054_) {
                                                phi_30455_ = true;
                                            } else {
                                                phi_30455_ = (0f >= _e7054_);
                                            }
                                            let _e7073_ = phi_30455_;
                                            let _e7077_ = bitcast<f32>((bitcast<u32>(_e6754_) & 2147483647u));
                                            let _e7080_ = bitcast<f32>((bitcast<u32>(_e6759_) & 2147483647u));
                                            let _e7083_ = bitcast<f32>((bitcast<u32>(_e6764_) & 2147483647u));
                                            let _e7086_ = bitcast<f32>((bitcast<u32>(_e6769_) & 2147483647u));
                                            let _e7089_ = bitcast<f32>((bitcast<u32>(_e6814_) & 2147483647u));
                                            let _e7092_ = bitcast<f32>((bitcast<u32>(_e6819_) & 2147483647u));
                                            let _e7095_ = bitcast<f32>((bitcast<u32>(_e6824_) & 2147483647u));
                                            let _e7098_ = bitcast<f32>((bitcast<u32>(_e6829_) & 2147483647u));
                                            if (_e7077_ != _e7077_) {
                                                phi_30470_ = true;
                                            } else {
                                                phi_30470_ = (_e7089_ >= _e7077_);
                                            }
                                            let _e7102_ = phi_30470_;
                                            let _e7103_ = select(_e7077_, _e7089_, _e7102_);
                                            if (_e7080_ != _e7080_) {
                                                phi_30485_ = true;
                                            } else {
                                                phi_30485_ = (_e7092_ >= _e7080_);
                                            }
                                            let _e7107_ = phi_30485_;
                                            let _e7108_ = select(_e7080_, _e7092_, _e7107_);
                                            if (_e7083_ != _e7083_) {
                                                phi_30500_ = true;
                                            } else {
                                                phi_30500_ = (_e7095_ >= _e7083_);
                                            }
                                            let _e7112_ = phi_30500_;
                                            let _e7113_ = select(_e7083_, _e7095_, _e7112_);
                                            if (_e7086_ != _e7086_) {
                                                phi_30515_ = true;
                                            } else {
                                                phi_30515_ = (_e7098_ >= _e7086_);
                                            }
                                            let _e7117_ = phi_30515_;
                                            let _e7118_ = select(_e7086_, _e7098_, _e7117_);
                                            let _e7135_ = (_e7103_ <= 0f);
                                            let _e7136_ = select((_e6951_ * (1.5f - (0.5f * (_e6963_ + _e6967_)))), 0f, _e7135_);
                                            let _e7137_ = (_e7108_ <= 0f);
                                            let _e7138_ = select((_e6952_ * (1.5f - (0.5f * (_e6964_ + _e6968_)))), 0f, _e7137_);
                                            let _e7139_ = (_e7113_ <= 0f);
                                            let _e7140_ = select((_e6953_ * (1.5f - (0.5f * (_e6965_ + _e6969_)))), 0f, _e7139_);
                                            let _e7141_ = (_e7118_ <= 0f);
                                            let _e7142_ = select((_e6954_ * (1.5f - (0.5f * (_e6966_ + _e6970_)))), 0f, _e7141_);
                                            let _e7184_ = select(((_e6963_ + (0.33333334f * _e6967_)) - (((_e6963_ * _e6963_) + ((2f * _e6963_) * _e6967_)) + (0.2f * (_e6967_ * _e6967_)))), 0f, _e7135_);
                                            let _e7185_ = select(((_e6964_ + (0.33333334f * _e6968_)) - (((_e6964_ * _e6964_) + ((2f * _e6964_) * _e6968_)) + (0.2f * (_e6968_ * _e6968_)))), 0f, _e7137_);
                                            let _e7186_ = select(((_e6965_ + (0.33333334f * _e6969_)) - (((_e6965_ * _e6965_) + ((2f * _e6965_) * _e6969_)) + (0.2f * (_e6969_ * _e6969_)))), 0f, _e7139_);
                                            let _e7187_ = select(((_e6966_ + (0.33333334f * _e6970_)) - (((_e6966_ * _e6966_) + ((2f * _e6966_) * _e6970_)) + (0.2f * (_e6970_ * _e6970_)))), 0f, _e7141_);
                                            if (_e7103_ <= 1f) {
                                                if (_e7108_ <= 1f) {
                                                    let _e7191_ = (_e7113_ <= 1f);
                                                    if _e7191_ {
                                                        phi_19453_ = (_e7118_ <= 1f);
                                                    } else {
                                                        phi_19453_ = bool();
                                                    }
                                                    let _e7194_ = phi_19453_;
                                                    phi_19455_ = _e7194_;
                                                    phi_19456_ = select(true, false, _e7191_);
                                                } else {
                                                    phi_19455_ = bool();
                                                    phi_19456_ = true;
                                                }
                                                let _e7197_ = phi_19455_;
                                                let _e7199_ = phi_19456_;
                                                phi_19457_ = _e7197_;
                                                phi_19458_ = _e7199_;
                                            } else {
                                                phi_19457_ = bool();
                                                phi_19458_ = true;
                                            }
                                            let _e7201_ = phi_19457_;
                                            let _e7203_ = phi_19458_;
                                            if select(_e7201_, false, _e7203_) {
                                                phi_19836_ = type_37_(type_30_(array<f32, 4>(_e7136_, _e7138_, _e7140_, _e7142_)), type_30_(array<f32, 4>(_e7184_, _e7185_, _e7186_, _e7187_)), type_30_(_e6830_));
                                            } else {
                                                if (_e6991_ != _e6991_) {
                                                    phi_30530_ = true;
                                                } else {
                                                    phi_30530_ = (0.000001f >= _e6991_);
                                                }
                                                let _e7208_ = phi_30530_;
                                                let _e7209_ = select(_e6991_, 0.000001f, _e7208_);
                                                if (_e6992_ != _e6992_) {
                                                    phi_30545_ = true;
                                                } else {
                                                    phi_30545_ = (0.000001f >= _e6992_);
                                                }
                                                let _e7213_ = phi_30545_;
                                                let _e7214_ = select(_e6992_, 0.000001f, _e7213_);
                                                if (_e6993_ != _e6993_) {
                                                    phi_30560_ = true;
                                                } else {
                                                    phi_30560_ = (0.000001f >= _e6993_);
                                                }
                                                let _e7218_ = phi_30560_;
                                                let _e7219_ = select(_e6993_, 0.000001f, _e7218_);
                                                if (_e6994_ != _e6994_) {
                                                    phi_30575_ = true;
                                                } else {
                                                    phi_30575_ = (0.000001f >= _e6994_);
                                                }
                                                let _e7223_ = phi_30575_;
                                                let _e7224_ = select(_e6994_, 0.000001f, _e7223_);
                                                let _e7227_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7209_)));
                                                let _e7230_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7214_)));
                                                let _e7233_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7219_)));
                                                let _e7236_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7224_)));
                                                let _e7246_ = (_e7227_ + (_e7227_ * (1f - (_e7209_ * _e7227_))));
                                                let _e7248_ = (_e7230_ + (_e7230_ * (1f - (_e7214_ * _e7230_))));
                                                let _e7250_ = (_e7233_ + (_e7233_ * (1f - (_e7219_ * _e7233_))));
                                                let _e7252_ = (_e7236_ + (_e7236_ * (1f - (_e7224_ * _e7236_))));
                                                let _e7262_ = (_e7246_ + (_e7246_ * (1f - (_e7209_ * _e7246_))));
                                                let _e7264_ = (_e7248_ + (_e7248_ * (1f - (_e7214_ * _e7248_))));
                                                let _e7266_ = (_e7250_ + (_e7250_ * (1f - (_e7219_ * _e7250_))));
                                                let _e7268_ = (_e7252_ + (_e7252_ * (1f - (_e7224_ * _e7252_))));
                                                let _e7321_ = (select((select(0f, 1f, (-1f <= _e6835_)) * select(0f, 1f, (_e6854_ <= 1f))), 1f, (_e6991_ <= 0.000001f)) <= 0f);
                                                let _e7323_ = (select((select(0f, 1f, (-1f <= _e6840_)) * select(0f, 1f, (_e6858_ <= 1f))), 1f, (_e6992_ <= 0.000001f)) <= 0f);
                                                let _e7325_ = (select((select(0f, 1f, (-1f <= _e6845_)) * select(0f, 1f, (_e6862_ <= 1f))), 1f, (_e6993_ <= 0.000001f)) <= 0f);
                                                let _e7327_ = (select((select(0f, 1f, (-1f <= _e6850_)) * select(0f, 1f, (_e6866_ <= 1f))), 1f, (_e6994_ <= 0.000001f)) <= 0f);
                                                let _e7333_ = (1f <= _e6835_);
                                                let _e7335_ = (1f <= _e6840_);
                                                let _e7337_ = (1f <= _e6845_);
                                                let _e7339_ = (1f <= _e6850_);
                                                let _e7341_ = (_e6854_ <= -1f);
                                                let _e7343_ = (_e6858_ <= -1f);
                                                let _e7345_ = (_e6862_ <= -1f);
                                                let _e7347_ = (_e6866_ <= -1f);
                                                phi_19836_ = type_37_(type_30_(array<f32, 4>(select(select(select(_e7136_, (((select(_e7051_, 0f, _e7058_) - select(_e7011_, 0f, _e7018_)) + (_e6975_ * _e7136_)) * _e7262_), _e7321_), 1f, _e7333_), -1f, _e7341_), select(select(select(_e7138_, (((select(_e7052_, 0f, _e7063_) - select(_e7012_, 0f, _e7023_)) + (_e6980_ * _e7138_)) * _e7264_), _e7323_), 1f, _e7335_), -1f, _e7343_), select(select(select(_e7140_, (((select(_e7053_, 0f, _e7068_) - select(_e7013_, 0f, _e7028_)) + (_e6985_ * _e7140_)) * _e7266_), _e7325_), 1f, _e7337_), -1f, _e7345_), select(select(select(_e7142_, (((select(_e7054_, 0f, _e7073_) - select(_e7014_, 0f, _e7033_)) + (_e6990_ * _e7142_)) * _e7268_), _e7327_), 1f, _e7339_), -1f, _e7347_))), type_30_(array<f32, 4>(select(select(select(_e7184_, ((_e6975_ * _e7184_) * _e7262_), _e7321_), 0f, _e7333_), 0f, _e7341_), select(select(select(_e7185_, ((_e6980_ * _e7185_) * _e7264_), _e7323_), 0f, _e7335_), 0f, _e7343_), select(select(select(_e7186_, ((_e6985_ * _e7186_) * _e7266_), _e7325_), 0f, _e7337_), 0f, _e7345_), select(select(select(_e7187_, ((_e6990_ * _e7187_) * _e7268_), _e7327_), 0f, _e7339_), 0f, _e7347_))), type_30_(_e6830_));
                                            }
                                            let _e7368_ = phi_19836_;
                                            let _e7383_ = select(0f, _e6700_[0], (bitcast<f32>((bitcast<u32>(_e6700_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7385_ = select(0f, _e6700_[1], (bitcast<f32>((bitcast<u32>(_e6700_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7387_ = select(0f, _e6700_[2], (bitcast<f32>((bitcast<u32>(_e6700_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7389_ = select(0f, _e6700_[3], (bitcast<f32>((bitcast<u32>(_e6700_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7390_ = (_e7383_ != _e7383_);
                                            if _e7390_ {
                                                phi_30590_ = true;
                                            } else {
                                                phi_30590_ = (-1000000000000000000f >= _e7383_);
                                            }
                                            let _e7393_ = phi_30590_;
                                            let _e7394_ = select(_e7383_, -1000000000000000000f, _e7393_);
                                            let _e7395_ = (_e7385_ != _e7385_);
                                            if _e7395_ {
                                                phi_30605_ = true;
                                            } else {
                                                phi_30605_ = (-1000000000000000000f >= _e7385_);
                                            }
                                            let _e7398_ = phi_30605_;
                                            let _e7399_ = select(_e7385_, -1000000000000000000f, _e7398_);
                                            let _e7400_ = (_e7387_ != _e7387_);
                                            if _e7400_ {
                                                phi_30620_ = true;
                                            } else {
                                                phi_30620_ = (-1000000000000000000f >= _e7387_);
                                            }
                                            let _e7403_ = phi_30620_;
                                            let _e7404_ = select(_e7387_, -1000000000000000000f, _e7403_);
                                            let _e7405_ = (_e7389_ != _e7389_);
                                            if _e7405_ {
                                                phi_30635_ = true;
                                            } else {
                                                phi_30635_ = (-1000000000000000000f >= _e7389_);
                                            }
                                            let _e7408_ = phi_30635_;
                                            let _e7409_ = select(_e7389_, -1000000000000000000f, _e7408_);
                                            if (_e7394_ != _e7394_) {
                                                phi_30650_ = true;
                                            } else {
                                                phi_30650_ = (1000000000000000000f <= _e7394_);
                                            }
                                            let _e7413_ = phi_30650_;
                                            let _e7414_ = select(_e7394_, 1000000000000000000f, _e7413_);
                                            if (_e7399_ != _e7399_) {
                                                phi_30665_ = true;
                                            } else {
                                                phi_30665_ = (1000000000000000000f <= _e7399_);
                                            }
                                            let _e7418_ = phi_30665_;
                                            let _e7419_ = select(_e7399_, 1000000000000000000f, _e7418_);
                                            if (_e7404_ != _e7404_) {
                                                phi_30680_ = true;
                                            } else {
                                                phi_30680_ = (1000000000000000000f <= _e7404_);
                                            }
                                            let _e7423_ = phi_30680_;
                                            let _e7424_ = select(_e7404_, 1000000000000000000f, _e7423_);
                                            if (_e7409_ != _e7409_) {
                                                phi_30695_ = true;
                                            } else {
                                                phi_30695_ = (1000000000000000000f <= _e7409_);
                                            }
                                            let _e7428_ = phi_30695_;
                                            let _e7429_ = select(_e7409_, 1000000000000000000f, _e7428_);
                                            let _e7443_ = select(0f, _e6705_[0], (bitcast<f32>((bitcast<u32>(_e6705_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7445_ = select(0f, _e6705_[1], (bitcast<f32>((bitcast<u32>(_e6705_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7447_ = select(0f, _e6705_[2], (bitcast<f32>((bitcast<u32>(_e6705_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7449_ = select(0f, _e6705_[3], (bitcast<f32>((bitcast<u32>(_e6705_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7450_ = (_e7443_ != _e7443_);
                                            if _e7450_ {
                                                phi_30710_ = true;
                                            } else {
                                                phi_30710_ = (-1000000000000000000f >= _e7443_);
                                            }
                                            let _e7453_ = phi_30710_;
                                            let _e7454_ = select(_e7443_, -1000000000000000000f, _e7453_);
                                            let _e7455_ = (_e7445_ != _e7445_);
                                            if _e7455_ {
                                                phi_30725_ = true;
                                            } else {
                                                phi_30725_ = (-1000000000000000000f >= _e7445_);
                                            }
                                            let _e7458_ = phi_30725_;
                                            let _e7459_ = select(_e7445_, -1000000000000000000f, _e7458_);
                                            let _e7460_ = (_e7447_ != _e7447_);
                                            if _e7460_ {
                                                phi_30740_ = true;
                                            } else {
                                                phi_30740_ = (-1000000000000000000f >= _e7447_);
                                            }
                                            let _e7463_ = phi_30740_;
                                            let _e7464_ = select(_e7447_, -1000000000000000000f, _e7463_);
                                            let _e7465_ = (_e7449_ != _e7449_);
                                            if _e7465_ {
                                                phi_30755_ = true;
                                            } else {
                                                phi_30755_ = (-1000000000000000000f >= _e7449_);
                                            }
                                            let _e7468_ = phi_30755_;
                                            let _e7469_ = select(_e7449_, -1000000000000000000f, _e7468_);
                                            if (_e7454_ != _e7454_) {
                                                phi_30770_ = true;
                                            } else {
                                                phi_30770_ = (1000000000000000000f <= _e7454_);
                                            }
                                            let _e7473_ = phi_30770_;
                                            let _e7474_ = select(_e7454_, 1000000000000000000f, _e7473_);
                                            if (_e7459_ != _e7459_) {
                                                phi_30785_ = true;
                                            } else {
                                                phi_30785_ = (1000000000000000000f <= _e7459_);
                                            }
                                            let _e7478_ = phi_30785_;
                                            let _e7479_ = select(_e7459_, 1000000000000000000f, _e7478_);
                                            if (_e7464_ != _e7464_) {
                                                phi_30800_ = true;
                                            } else {
                                                phi_30800_ = (1000000000000000000f <= _e7464_);
                                            }
                                            let _e7483_ = phi_30800_;
                                            let _e7484_ = select(_e7464_, 1000000000000000000f, _e7483_);
                                            if (_e7469_ != _e7469_) {
                                                phi_30815_ = true;
                                            } else {
                                                phi_30815_ = (1000000000000000000f <= _e7469_);
                                            }
                                            let _e7488_ = phi_30815_;
                                            let _e7489_ = select(_e7469_, 1000000000000000000f, _e7488_);
                                            let _e7490_ = array<f32, 4>(_e7474_, _e7479_, _e7484_, _e7489_);
                                            let _e7491_ = (_e7414_ != _e7414_);
                                            if _e7491_ {
                                                phi_30830_ = true;
                                            } else {
                                                phi_30830_ = (_e7474_ <= _e7414_);
                                            }
                                            let _e7494_ = phi_30830_;
                                            let _e7495_ = select(_e7414_, _e7474_, _e7494_);
                                            let _e7496_ = (_e7419_ != _e7419_);
                                            if _e7496_ {
                                                phi_30845_ = true;
                                            } else {
                                                phi_30845_ = (_e7479_ <= _e7419_);
                                            }
                                            let _e7499_ = phi_30845_;
                                            let _e7500_ = select(_e7419_, _e7479_, _e7499_);
                                            let _e7501_ = (_e7424_ != _e7424_);
                                            if _e7501_ {
                                                phi_30860_ = true;
                                            } else {
                                                phi_30860_ = (_e7484_ <= _e7424_);
                                            }
                                            let _e7504_ = phi_30860_;
                                            let _e7505_ = select(_e7424_, _e7484_, _e7504_);
                                            let _e7506_ = (_e7429_ != _e7429_);
                                            if _e7506_ {
                                                phi_30875_ = true;
                                            } else {
                                                phi_30875_ = (_e7489_ <= _e7429_);
                                            }
                                            let _e7509_ = phi_30875_;
                                            let _e7510_ = select(_e7429_, _e7489_, _e7509_);
                                            if _e7491_ {
                                                phi_30890_ = true;
                                            } else {
                                                phi_30890_ = (_e7474_ >= _e7414_);
                                            }
                                            let _e7513_ = phi_30890_;
                                            let _e7514_ = select(_e7414_, _e7474_, _e7513_);
                                            if _e7496_ {
                                                phi_30905_ = true;
                                            } else {
                                                phi_30905_ = (_e7479_ >= _e7419_);
                                            }
                                            let _e7517_ = phi_30905_;
                                            let _e7518_ = select(_e7419_, _e7479_, _e7517_);
                                            if _e7501_ {
                                                phi_30920_ = true;
                                            } else {
                                                phi_30920_ = (_e7484_ >= _e7424_);
                                            }
                                            let _e7521_ = phi_30920_;
                                            let _e7522_ = select(_e7424_, _e7484_, _e7521_);
                                            if _e7506_ {
                                                phi_30935_ = true;
                                            } else {
                                                phi_30935_ = (_e7489_ >= _e7429_);
                                            }
                                            let _e7525_ = phi_30935_;
                                            let _e7526_ = select(_e7429_, _e7489_, _e7525_);
                                            let _e7527_ = (_e7495_ != _e7495_);
                                            if _e7527_ {
                                                phi_30950_ = true;
                                            } else {
                                                phi_30950_ = (-1f >= _e7495_);
                                            }
                                            let _e7530_ = phi_30950_;
                                            let _e7531_ = select(_e7495_, -1f, _e7530_);
                                            let _e7532_ = (_e7500_ != _e7500_);
                                            if _e7532_ {
                                                phi_30965_ = true;
                                            } else {
                                                phi_30965_ = (-1f >= _e7500_);
                                            }
                                            let _e7535_ = phi_30965_;
                                            let _e7536_ = select(_e7500_, -1f, _e7535_);
                                            let _e7537_ = (_e7505_ != _e7505_);
                                            if _e7537_ {
                                                phi_30980_ = true;
                                            } else {
                                                phi_30980_ = (-1f >= _e7505_);
                                            }
                                            let _e7540_ = phi_30980_;
                                            let _e7541_ = select(_e7505_, -1f, _e7540_);
                                            let _e7542_ = (_e7510_ != _e7510_);
                                            if _e7542_ {
                                                phi_30995_ = true;
                                            } else {
                                                phi_30995_ = (-1f >= _e7510_);
                                            }
                                            let _e7545_ = phi_30995_;
                                            let _e7546_ = select(_e7510_, -1f, _e7545_);
                                            if (_e7531_ != _e7531_) {
                                                phi_31010_ = true;
                                            } else {
                                                phi_31010_ = (1f <= _e7531_);
                                            }
                                            let _e7550_ = phi_31010_;
                                            let _e7551_ = select(_e7531_, 1f, _e7550_);
                                            if (_e7536_ != _e7536_) {
                                                phi_31025_ = true;
                                            } else {
                                                phi_31025_ = (1f <= _e7536_);
                                            }
                                            let _e7555_ = phi_31025_;
                                            let _e7556_ = select(_e7536_, 1f, _e7555_);
                                            if (_e7541_ != _e7541_) {
                                                phi_31040_ = true;
                                            } else {
                                                phi_31040_ = (1f <= _e7541_);
                                            }
                                            let _e7560_ = phi_31040_;
                                            let _e7561_ = select(_e7541_, 1f, _e7560_);
                                            if (_e7546_ != _e7546_) {
                                                phi_31055_ = true;
                                            } else {
                                                phi_31055_ = (1f <= _e7546_);
                                            }
                                            let _e7565_ = phi_31055_;
                                            let _e7566_ = select(_e7546_, 1f, _e7565_);
                                            let _e7567_ = (_e7514_ != _e7514_);
                                            if _e7567_ {
                                                phi_31070_ = true;
                                            } else {
                                                phi_31070_ = (-1f >= _e7514_);
                                            }
                                            let _e7570_ = phi_31070_;
                                            let _e7571_ = select(_e7514_, -1f, _e7570_);
                                            let _e7572_ = (_e7518_ != _e7518_);
                                            if _e7572_ {
                                                phi_31085_ = true;
                                            } else {
                                                phi_31085_ = (-1f >= _e7518_);
                                            }
                                            let _e7575_ = phi_31085_;
                                            let _e7576_ = select(_e7518_, -1f, _e7575_);
                                            let _e7577_ = (_e7522_ != _e7522_);
                                            if _e7577_ {
                                                phi_31100_ = true;
                                            } else {
                                                phi_31100_ = (-1f >= _e7522_);
                                            }
                                            let _e7580_ = phi_31100_;
                                            let _e7581_ = select(_e7522_, -1f, _e7580_);
                                            let _e7582_ = (_e7526_ != _e7526_);
                                            if _e7582_ {
                                                phi_31115_ = true;
                                            } else {
                                                phi_31115_ = (-1f >= _e7526_);
                                            }
                                            let _e7585_ = phi_31115_;
                                            let _e7586_ = select(_e7526_, -1f, _e7585_);
                                            if (_e7571_ != _e7571_) {
                                                phi_31130_ = true;
                                            } else {
                                                phi_31130_ = (1f <= _e7571_);
                                            }
                                            let _e7590_ = phi_31130_;
                                            let _e7591_ = select(_e7571_, 1f, _e7590_);
                                            if (_e7576_ != _e7576_) {
                                                phi_31145_ = true;
                                            } else {
                                                phi_31145_ = (1f <= _e7576_);
                                            }
                                            let _e7595_ = phi_31145_;
                                            let _e7596_ = select(_e7576_, 1f, _e7595_);
                                            if (_e7581_ != _e7581_) {
                                                phi_31160_ = true;
                                            } else {
                                                phi_31160_ = (1f <= _e7581_);
                                            }
                                            let _e7600_ = phi_31160_;
                                            let _e7601_ = select(_e7581_, 1f, _e7600_);
                                            if (_e7586_ != _e7586_) {
                                                phi_31175_ = true;
                                            } else {
                                                phi_31175_ = (1f <= _e7586_);
                                            }
                                            let _e7605_ = phi_31175_;
                                            let _e7606_ = select(_e7586_, 1f, _e7605_);
                                            let _e7611_ = (0.5f * (_e7591_ + _e7551_));
                                            let _e7612_ = (0.5f * (_e7596_ + _e7556_));
                                            let _e7613_ = (0.5f * (_e7601_ + _e7561_));
                                            let _e7614_ = (0.5f * (_e7606_ + _e7566_));
                                            let _e7615_ = (_e7591_ - _e7551_);
                                            let _e7616_ = (_e7596_ - _e7556_);
                                            let _e7617_ = (_e7601_ - _e7561_);
                                            let _e7618_ = (_e7606_ - _e7566_);
                                            let _e7619_ = (0.5f * _e7615_);
                                            let _e7620_ = (0.5f * _e7616_);
                                            let _e7621_ = (0.5f * _e7617_);
                                            let _e7622_ = (0.5f * _e7618_);
                                            let _e7623_ = (_e7611_ * _e7611_);
                                            let _e7624_ = (_e7612_ * _e7612_);
                                            let _e7625_ = (_e7613_ * _e7613_);
                                            let _e7626_ = (_e7614_ * _e7614_);
                                            let _e7627_ = (_e7619_ * _e7619_);
                                            let _e7628_ = (_e7620_ * _e7620_);
                                            let _e7629_ = (_e7621_ * _e7621_);
                                            let _e7630_ = (_e7622_ * _e7622_);
                                            if (_e7615_ != _e7615_) {
                                                phi_31190_ = true;
                                            } else {
                                                phi_31190_ = (0f >= _e7615_);
                                            }
                                            let _e7634_ = phi_31190_;
                                            let _e7635_ = select(_e7615_, 0f, _e7634_);
                                            if (_e7616_ != _e7616_) {
                                                phi_31205_ = true;
                                            } else {
                                                phi_31205_ = (0f >= _e7616_);
                                            }
                                            let _e7639_ = phi_31205_;
                                            let _e7640_ = select(_e7616_, 0f, _e7639_);
                                            if (_e7617_ != _e7617_) {
                                                phi_31220_ = true;
                                            } else {
                                                phi_31220_ = (0f >= _e7617_);
                                            }
                                            let _e7644_ = phi_31220_;
                                            let _e7645_ = select(_e7617_, 0f, _e7644_);
                                            if (_e7618_ != _e7618_) {
                                                phi_31235_ = true;
                                            } else {
                                                phi_31235_ = (0f >= _e7618_);
                                            }
                                            let _e7649_ = phi_31235_;
                                            let _e7650_ = select(_e7618_, 0f, _e7649_);
                                            let _e7651_ = (_e7514_ - _e7495_);
                                            let _e7652_ = (_e7518_ - _e7500_);
                                            let _e7653_ = (_e7522_ - _e7505_);
                                            let _e7654_ = (_e7526_ - _e7510_);
                                            if _e7567_ {
                                                phi_31250_ = true;
                                            } else {
                                                phi_31250_ = (-1f <= _e7514_);
                                            }
                                            let _e7657_ = phi_31250_;
                                            if _e7572_ {
                                                phi_31265_ = true;
                                            } else {
                                                phi_31265_ = (-1f <= _e7518_);
                                            }
                                            let _e7661_ = phi_31265_;
                                            if _e7577_ {
                                                phi_31280_ = true;
                                            } else {
                                                phi_31280_ = (-1f <= _e7522_);
                                            }
                                            let _e7665_ = phi_31280_;
                                            if _e7582_ {
                                                phi_31295_ = true;
                                            } else {
                                                phi_31295_ = (-1f <= _e7526_);
                                            }
                                            let _e7669_ = phi_31295_;
                                            let _e7671_ = (select(_e7514_, -1f, _e7657_) - _e7495_);
                                            let _e7672_ = (select(_e7518_, -1f, _e7661_) - _e7500_);
                                            let _e7673_ = (select(_e7522_, -1f, _e7665_) - _e7505_);
                                            let _e7674_ = (select(_e7526_, -1f, _e7669_) - _e7510_);
                                            if (_e7671_ != _e7671_) {
                                                phi_31310_ = true;
                                            } else {
                                                phi_31310_ = (0f >= _e7671_);
                                            }
                                            let _e7678_ = phi_31310_;
                                            if (_e7672_ != _e7672_) {
                                                phi_31325_ = true;
                                            } else {
                                                phi_31325_ = (0f >= _e7672_);
                                            }
                                            let _e7683_ = phi_31325_;
                                            if (_e7673_ != _e7673_) {
                                                phi_31340_ = true;
                                            } else {
                                                phi_31340_ = (0f >= _e7673_);
                                            }
                                            let _e7688_ = phi_31340_;
                                            if (_e7674_ != _e7674_) {
                                                phi_31355_ = true;
                                            } else {
                                                phi_31355_ = (0f >= _e7674_);
                                            }
                                            let _e7693_ = phi_31355_;
                                            if _e7527_ {
                                                phi_31370_ = true;
                                            } else {
                                                phi_31370_ = (1f >= _e7495_);
                                            }
                                            let _e7697_ = phi_31370_;
                                            if _e7532_ {
                                                phi_31385_ = true;
                                            } else {
                                                phi_31385_ = (1f >= _e7500_);
                                            }
                                            let _e7701_ = phi_31385_;
                                            if _e7537_ {
                                                phi_31400_ = true;
                                            } else {
                                                phi_31400_ = (1f >= _e7505_);
                                            }
                                            let _e7705_ = phi_31400_;
                                            if _e7542_ {
                                                phi_31415_ = true;
                                            } else {
                                                phi_31415_ = (1f >= _e7510_);
                                            }
                                            let _e7709_ = phi_31415_;
                                            let _e7711_ = (_e7514_ - select(_e7495_, 1f, _e7697_));
                                            let _e7712_ = (_e7518_ - select(_e7500_, 1f, _e7701_));
                                            let _e7713_ = (_e7522_ - select(_e7505_, 1f, _e7705_));
                                            let _e7714_ = (_e7526_ - select(_e7510_, 1f, _e7709_));
                                            if (_e7711_ != _e7711_) {
                                                phi_31430_ = true;
                                            } else {
                                                phi_31430_ = (0f >= _e7711_);
                                            }
                                            let _e7718_ = phi_31430_;
                                            if (_e7712_ != _e7712_) {
                                                phi_31445_ = true;
                                            } else {
                                                phi_31445_ = (0f >= _e7712_);
                                            }
                                            let _e7723_ = phi_31445_;
                                            if (_e7713_ != _e7713_) {
                                                phi_31460_ = true;
                                            } else {
                                                phi_31460_ = (0f >= _e7713_);
                                            }
                                            let _e7728_ = phi_31460_;
                                            if (_e7714_ != _e7714_) {
                                                phi_31475_ = true;
                                            } else {
                                                phi_31475_ = (0f >= _e7714_);
                                            }
                                            let _e7733_ = phi_31475_;
                                            let _e7737_ = bitcast<f32>((bitcast<u32>(_e7414_) & 2147483647u));
                                            let _e7740_ = bitcast<f32>((bitcast<u32>(_e7419_) & 2147483647u));
                                            let _e7743_ = bitcast<f32>((bitcast<u32>(_e7424_) & 2147483647u));
                                            let _e7746_ = bitcast<f32>((bitcast<u32>(_e7429_) & 2147483647u));
                                            let _e7749_ = bitcast<f32>((bitcast<u32>(_e7474_) & 2147483647u));
                                            let _e7752_ = bitcast<f32>((bitcast<u32>(_e7479_) & 2147483647u));
                                            let _e7755_ = bitcast<f32>((bitcast<u32>(_e7484_) & 2147483647u));
                                            let _e7758_ = bitcast<f32>((bitcast<u32>(_e7489_) & 2147483647u));
                                            if (_e7737_ != _e7737_) {
                                                phi_31490_ = true;
                                            } else {
                                                phi_31490_ = (_e7749_ >= _e7737_);
                                            }
                                            let _e7762_ = phi_31490_;
                                            let _e7763_ = select(_e7737_, _e7749_, _e7762_);
                                            if (_e7740_ != _e7740_) {
                                                phi_31505_ = true;
                                            } else {
                                                phi_31505_ = (_e7752_ >= _e7740_);
                                            }
                                            let _e7767_ = phi_31505_;
                                            let _e7768_ = select(_e7740_, _e7752_, _e7767_);
                                            if (_e7743_ != _e7743_) {
                                                phi_31520_ = true;
                                            } else {
                                                phi_31520_ = (_e7755_ >= _e7743_);
                                            }
                                            let _e7772_ = phi_31520_;
                                            let _e7773_ = select(_e7743_, _e7755_, _e7772_);
                                            if (_e7746_ != _e7746_) {
                                                phi_31535_ = true;
                                            } else {
                                                phi_31535_ = (_e7758_ >= _e7746_);
                                            }
                                            let _e7777_ = phi_31535_;
                                            let _e7778_ = select(_e7746_, _e7758_, _e7777_);
                                            let _e7795_ = (_e7763_ <= 0f);
                                            let _e7796_ = select((_e7611_ * (1.5f - (0.5f * (_e7623_ + _e7627_)))), 0f, _e7795_);
                                            let _e7797_ = (_e7768_ <= 0f);
                                            let _e7798_ = select((_e7612_ * (1.5f - (0.5f * (_e7624_ + _e7628_)))), 0f, _e7797_);
                                            let _e7799_ = (_e7773_ <= 0f);
                                            let _e7800_ = select((_e7613_ * (1.5f - (0.5f * (_e7625_ + _e7629_)))), 0f, _e7799_);
                                            let _e7801_ = (_e7778_ <= 0f);
                                            let _e7802_ = select((_e7614_ * (1.5f - (0.5f * (_e7626_ + _e7630_)))), 0f, _e7801_);
                                            let _e7844_ = select(((_e7623_ + (0.33333334f * _e7627_)) - (((_e7623_ * _e7623_) + ((2f * _e7623_) * _e7627_)) + (0.2f * (_e7627_ * _e7627_)))), 0f, _e7795_);
                                            let _e7845_ = select(((_e7624_ + (0.33333334f * _e7628_)) - (((_e7624_ * _e7624_) + ((2f * _e7624_) * _e7628_)) + (0.2f * (_e7628_ * _e7628_)))), 0f, _e7797_);
                                            let _e7846_ = select(((_e7625_ + (0.33333334f * _e7629_)) - (((_e7625_ * _e7625_) + ((2f * _e7625_) * _e7629_)) + (0.2f * (_e7629_ * _e7629_)))), 0f, _e7799_);
                                            let _e7847_ = select(((_e7626_ + (0.33333334f * _e7630_)) - (((_e7626_ * _e7626_) + ((2f * _e7626_) * _e7630_)) + (0.2f * (_e7630_ * _e7630_)))), 0f, _e7801_);
                                            if (_e7763_ <= 1f) {
                                                if (_e7768_ <= 1f) {
                                                    let _e7851_ = (_e7773_ <= 1f);
                                                    if _e7851_ {
                                                        phi_20282_ = (_e7778_ <= 1f);
                                                    } else {
                                                        phi_20282_ = bool();
                                                    }
                                                    let _e7854_ = phi_20282_;
                                                    phi_20284_ = _e7854_;
                                                    phi_20285_ = select(true, false, _e7851_);
                                                } else {
                                                    phi_20284_ = bool();
                                                    phi_20285_ = true;
                                                }
                                                let _e7857_ = phi_20284_;
                                                let _e7859_ = phi_20285_;
                                                phi_20286_ = _e7857_;
                                                phi_20287_ = _e7859_;
                                            } else {
                                                phi_20286_ = bool();
                                                phi_20287_ = true;
                                            }
                                            let _e7861_ = phi_20286_;
                                            let _e7863_ = phi_20287_;
                                            if select(_e7861_, false, _e7863_) {
                                                phi_20665_ = type_37_(type_30_(array<f32, 4>(_e7796_, _e7798_, _e7800_, _e7802_)), type_30_(array<f32, 4>(_e7844_, _e7845_, _e7846_, _e7847_)), type_30_(_e7490_));
                                            } else {
                                                if (_e7651_ != _e7651_) {
                                                    phi_31550_ = true;
                                                } else {
                                                    phi_31550_ = (0.000001f >= _e7651_);
                                                }
                                                let _e7868_ = phi_31550_;
                                                let _e7869_ = select(_e7651_, 0.000001f, _e7868_);
                                                if (_e7652_ != _e7652_) {
                                                    phi_31565_ = true;
                                                } else {
                                                    phi_31565_ = (0.000001f >= _e7652_);
                                                }
                                                let _e7873_ = phi_31565_;
                                                let _e7874_ = select(_e7652_, 0.000001f, _e7873_);
                                                if (_e7653_ != _e7653_) {
                                                    phi_31580_ = true;
                                                } else {
                                                    phi_31580_ = (0.000001f >= _e7653_);
                                                }
                                                let _e7878_ = phi_31580_;
                                                let _e7879_ = select(_e7653_, 0.000001f, _e7878_);
                                                if (_e7654_ != _e7654_) {
                                                    phi_31595_ = true;
                                                } else {
                                                    phi_31595_ = (0.000001f >= _e7654_);
                                                }
                                                let _e7883_ = phi_31595_;
                                                let _e7884_ = select(_e7654_, 0.000001f, _e7883_);
                                                let _e7887_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7869_)));
                                                let _e7890_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7874_)));
                                                let _e7893_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7879_)));
                                                let _e7896_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7884_)));
                                                let _e7906_ = (_e7887_ + (_e7887_ * (1f - (_e7869_ * _e7887_))));
                                                let _e7908_ = (_e7890_ + (_e7890_ * (1f - (_e7874_ * _e7890_))));
                                                let _e7910_ = (_e7893_ + (_e7893_ * (1f - (_e7879_ * _e7893_))));
                                                let _e7912_ = (_e7896_ + (_e7896_ * (1f - (_e7884_ * _e7896_))));
                                                let _e7922_ = (_e7906_ + (_e7906_ * (1f - (_e7869_ * _e7906_))));
                                                let _e7924_ = (_e7908_ + (_e7908_ * (1f - (_e7874_ * _e7908_))));
                                                let _e7926_ = (_e7910_ + (_e7910_ * (1f - (_e7879_ * _e7910_))));
                                                let _e7928_ = (_e7912_ + (_e7912_ * (1f - (_e7884_ * _e7912_))));
                                                let _e7981_ = (select((select(0f, 1f, (-1f <= _e7495_)) * select(0f, 1f, (_e7514_ <= 1f))), 1f, (_e7651_ <= 0.000001f)) <= 0f);
                                                let _e7983_ = (select((select(0f, 1f, (-1f <= _e7500_)) * select(0f, 1f, (_e7518_ <= 1f))), 1f, (_e7652_ <= 0.000001f)) <= 0f);
                                                let _e7985_ = (select((select(0f, 1f, (-1f <= _e7505_)) * select(0f, 1f, (_e7522_ <= 1f))), 1f, (_e7653_ <= 0.000001f)) <= 0f);
                                                let _e7987_ = (select((select(0f, 1f, (-1f <= _e7510_)) * select(0f, 1f, (_e7526_ <= 1f))), 1f, (_e7654_ <= 0.000001f)) <= 0f);
                                                let _e7993_ = (1f <= _e7495_);
                                                let _e7995_ = (1f <= _e7500_);
                                                let _e7997_ = (1f <= _e7505_);
                                                let _e7999_ = (1f <= _e7510_);
                                                let _e8001_ = (_e7514_ <= -1f);
                                                let _e8003_ = (_e7518_ <= -1f);
                                                let _e8005_ = (_e7522_ <= -1f);
                                                let _e8007_ = (_e7526_ <= -1f);
                                                phi_20665_ = type_37_(type_30_(array<f32, 4>(select(select(select(_e7796_, (((select(_e7711_, 0f, _e7718_) - select(_e7671_, 0f, _e7678_)) + (_e7635_ * _e7796_)) * _e7922_), _e7981_), 1f, _e7993_), -1f, _e8001_), select(select(select(_e7798_, (((select(_e7712_, 0f, _e7723_) - select(_e7672_, 0f, _e7683_)) + (_e7640_ * _e7798_)) * _e7924_), _e7983_), 1f, _e7995_), -1f, _e8003_), select(select(select(_e7800_, (((select(_e7713_, 0f, _e7728_) - select(_e7673_, 0f, _e7688_)) + (_e7645_ * _e7800_)) * _e7926_), _e7985_), 1f, _e7997_), -1f, _e8005_), select(select(select(_e7802_, (((select(_e7714_, 0f, _e7733_) - select(_e7674_, 0f, _e7693_)) + (_e7650_ * _e7802_)) * _e7928_), _e7987_), 1f, _e7999_), -1f, _e8007_))), type_30_(array<f32, 4>(select(select(select(_e7844_, ((_e7635_ * _e7844_) * _e7922_), _e7981_), 0f, _e7993_), 0f, _e8001_), select(select(select(_e7845_, ((_e7640_ * _e7845_) * _e7924_), _e7983_), 0f, _e7995_), 0f, _e8003_), select(select(select(_e7846_, ((_e7645_ * _e7846_) * _e7926_), _e7985_), 0f, _e7997_), 0f, _e8005_), select(select(select(_e7847_, ((_e7650_ * _e7847_) * _e7928_), _e7987_), 0f, _e7999_), 0f, _e8007_))), type_30_(_e7490_));
                                            }
                                            let _e8028_ = phi_20665_;
                                            if (_e489_ <= 0f) {
                                                if (_e509_ <= 0f) {
                                                    let _e8032_ = (_e529_ <= 0f);
                                                    if _e8032_ {
                                                        phi_20682_ = (_e549_ <= 0f);
                                                    } else {
                                                        phi_20682_ = bool();
                                                    }
                                                    let _e8035_ = phi_20682_;
                                                    phi_20684_ = _e8035_;
                                                    phi_20685_ = select(true, false, _e8032_);
                                                } else {
                                                    phi_20684_ = bool();
                                                    phi_20685_ = true;
                                                }
                                                let _e8038_ = phi_20684_;
                                                let _e8040_ = phi_20685_;
                                                phi_20686_ = _e8038_;
                                                phi_20687_ = _e8040_;
                                            } else {
                                                phi_20686_ = bool();
                                                phi_20687_ = true;
                                            }
                                            let _e8042_ = phi_20686_;
                                            let _e8044_ = phi_20687_;
                                            if select(_e8042_, false, _e8044_) {
                                                phi_22595_ = _e8028_.member;
                                                phi_22596_ = _e7368_.member;
                                            } else {
                                                if _e6730_ {
                                                    phi_31610_ = true;
                                                } else {
                                                    phi_31610_ = (-1000000000000000000f >= _e6723_);
                                                }
                                                let _e8048_ = phi_31610_;
                                                let _e8049_ = select(_e6723_, -1000000000000000000f, _e8048_);
                                                if _e6735_ {
                                                    phi_31625_ = true;
                                                } else {
                                                    phi_31625_ = (-1000000000000000000f >= _e6725_);
                                                }
                                                let _e8052_ = phi_31625_;
                                                let _e8053_ = select(_e6725_, -1000000000000000000f, _e8052_);
                                                if _e6740_ {
                                                    phi_31640_ = true;
                                                } else {
                                                    phi_31640_ = (-1000000000000000000f >= _e6727_);
                                                }
                                                let _e8056_ = phi_31640_;
                                                let _e8057_ = select(_e6727_, -1000000000000000000f, _e8056_);
                                                if _e6745_ {
                                                    phi_31655_ = true;
                                                } else {
                                                    phi_31655_ = (-1000000000000000000f >= _e6729_);
                                                }
                                                let _e8060_ = phi_31655_;
                                                let _e8061_ = select(_e6729_, -1000000000000000000f, _e8060_);
                                                if (_e8049_ != _e8049_) {
                                                    phi_31670_ = true;
                                                } else {
                                                    phi_31670_ = (1000000000000000000f <= _e8049_);
                                                }
                                                let _e8065_ = phi_31670_;
                                                let _e8066_ = select(_e8049_, 1000000000000000000f, _e8065_);
                                                if (_e8053_ != _e8053_) {
                                                    phi_31685_ = true;
                                                } else {
                                                    phi_31685_ = (1000000000000000000f <= _e8053_);
                                                }
                                                let _e8070_ = phi_31685_;
                                                let _e8071_ = select(_e8053_, 1000000000000000000f, _e8070_);
                                                if (_e8057_ != _e8057_) {
                                                    phi_31700_ = true;
                                                } else {
                                                    phi_31700_ = (1000000000000000000f <= _e8057_);
                                                }
                                                let _e8075_ = phi_31700_;
                                                let _e8076_ = select(_e8057_, 1000000000000000000f, _e8075_);
                                                if (_e8061_ != _e8061_) {
                                                    phi_31715_ = true;
                                                } else {
                                                    phi_31715_ = (1000000000000000000f <= _e8061_);
                                                }
                                                let _e8080_ = phi_31715_;
                                                let _e8081_ = select(_e8061_, 1000000000000000000f, _e8080_);
                                                if _e6790_ {
                                                    phi_31730_ = true;
                                                } else {
                                                    phi_31730_ = (-1000000000000000000f >= _e6783_);
                                                }
                                                let _e8084_ = phi_31730_;
                                                let _e8085_ = select(_e6783_, -1000000000000000000f, _e8084_);
                                                if _e6795_ {
                                                    phi_31745_ = true;
                                                } else {
                                                    phi_31745_ = (-1000000000000000000f >= _e6785_);
                                                }
                                                let _e8088_ = phi_31745_;
                                                let _e8089_ = select(_e6785_, -1000000000000000000f, _e8088_);
                                                if _e6800_ {
                                                    phi_31760_ = true;
                                                } else {
                                                    phi_31760_ = (-1000000000000000000f >= _e6787_);
                                                }
                                                let _e8092_ = phi_31760_;
                                                let _e8093_ = select(_e6787_, -1000000000000000000f, _e8092_);
                                                if _e6805_ {
                                                    phi_31775_ = true;
                                                } else {
                                                    phi_31775_ = (-1000000000000000000f >= _e6789_);
                                                }
                                                let _e8096_ = phi_31775_;
                                                let _e8097_ = select(_e6789_, -1000000000000000000f, _e8096_);
                                                if (_e8085_ != _e8085_) {
                                                    phi_31790_ = true;
                                                } else {
                                                    phi_31790_ = (1000000000000000000f <= _e8085_);
                                                }
                                                let _e8101_ = phi_31790_;
                                                let _e8102_ = select(_e8085_, 1000000000000000000f, _e8101_);
                                                if (_e8089_ != _e8089_) {
                                                    phi_31805_ = true;
                                                } else {
                                                    phi_31805_ = (1000000000000000000f <= _e8089_);
                                                }
                                                let _e8106_ = phi_31805_;
                                                let _e8107_ = select(_e8089_, 1000000000000000000f, _e8106_);
                                                if (_e8093_ != _e8093_) {
                                                    phi_31820_ = true;
                                                } else {
                                                    phi_31820_ = (1000000000000000000f <= _e8093_);
                                                }
                                                let _e8111_ = phi_31820_;
                                                let _e8112_ = select(_e8093_, 1000000000000000000f, _e8111_);
                                                if (_e8097_ != _e8097_) {
                                                    phi_31835_ = true;
                                                } else {
                                                    phi_31835_ = (1000000000000000000f <= _e8097_);
                                                }
                                                let _e8116_ = phi_31835_;
                                                let _e8117_ = select(_e8097_, 1000000000000000000f, _e8116_);
                                                let _e8118_ = (_e8066_ != _e8066_);
                                                if _e8118_ {
                                                    phi_31850_ = true;
                                                } else {
                                                    phi_31850_ = (_e8102_ <= _e8066_);
                                                }
                                                let _e8121_ = phi_31850_;
                                                let _e8122_ = select(_e8066_, _e8102_, _e8121_);
                                                let _e8123_ = (_e8071_ != _e8071_);
                                                if _e8123_ {
                                                    phi_31865_ = true;
                                                } else {
                                                    phi_31865_ = (_e8107_ <= _e8071_);
                                                }
                                                let _e8126_ = phi_31865_;
                                                let _e8127_ = select(_e8071_, _e8107_, _e8126_);
                                                let _e8128_ = (_e8076_ != _e8076_);
                                                if _e8128_ {
                                                    phi_31880_ = true;
                                                } else {
                                                    phi_31880_ = (_e8112_ <= _e8076_);
                                                }
                                                let _e8131_ = phi_31880_;
                                                let _e8132_ = select(_e8076_, _e8112_, _e8131_);
                                                let _e8133_ = (_e8081_ != _e8081_);
                                                if _e8133_ {
                                                    phi_31895_ = true;
                                                } else {
                                                    phi_31895_ = (_e8117_ <= _e8081_);
                                                }
                                                let _e8136_ = phi_31895_;
                                                let _e8137_ = select(_e8081_, _e8117_, _e8136_);
                                                if _e8118_ {
                                                    phi_31910_ = true;
                                                } else {
                                                    phi_31910_ = (_e8102_ >= _e8066_);
                                                }
                                                let _e8140_ = phi_31910_;
                                                let _e8141_ = select(_e8066_, _e8102_, _e8140_);
                                                if _e8123_ {
                                                    phi_31925_ = true;
                                                } else {
                                                    phi_31925_ = (_e8107_ >= _e8071_);
                                                }
                                                let _e8144_ = phi_31925_;
                                                let _e8145_ = select(_e8071_, _e8107_, _e8144_);
                                                if _e8128_ {
                                                    phi_31940_ = true;
                                                } else {
                                                    phi_31940_ = (_e8112_ >= _e8076_);
                                                }
                                                let _e8148_ = phi_31940_;
                                                let _e8149_ = select(_e8076_, _e8112_, _e8148_);
                                                if _e8133_ {
                                                    phi_31955_ = true;
                                                } else {
                                                    phi_31955_ = (_e8117_ >= _e8081_);
                                                }
                                                let _e8152_ = phi_31955_;
                                                let _e8153_ = select(_e8081_, _e8117_, _e8152_);
                                                if (_e8122_ != _e8122_) {
                                                    phi_31970_ = true;
                                                } else {
                                                    phi_31970_ = (-1f >= _e8122_);
                                                }
                                                let _e8157_ = phi_31970_;
                                                let _e8158_ = select(_e8122_, -1f, _e8157_);
                                                if (_e8127_ != _e8127_) {
                                                    phi_31985_ = true;
                                                } else {
                                                    phi_31985_ = (-1f >= _e8127_);
                                                }
                                                let _e8162_ = phi_31985_;
                                                let _e8163_ = select(_e8127_, -1f, _e8162_);
                                                if (_e8132_ != _e8132_) {
                                                    phi_32000_ = true;
                                                } else {
                                                    phi_32000_ = (-1f >= _e8132_);
                                                }
                                                let _e8167_ = phi_32000_;
                                                let _e8168_ = select(_e8132_, -1f, _e8167_);
                                                if (_e8137_ != _e8137_) {
                                                    phi_32015_ = true;
                                                } else {
                                                    phi_32015_ = (-1f >= _e8137_);
                                                }
                                                let _e8172_ = phi_32015_;
                                                let _e8173_ = select(_e8137_, -1f, _e8172_);
                                                if (_e8158_ != _e8158_) {
                                                    phi_32030_ = true;
                                                } else {
                                                    phi_32030_ = (1f <= _e8158_);
                                                }
                                                let _e8177_ = phi_32030_;
                                                let _e8178_ = select(_e8158_, 1f, _e8177_);
                                                if (_e8163_ != _e8163_) {
                                                    phi_32045_ = true;
                                                } else {
                                                    phi_32045_ = (1f <= _e8163_);
                                                }
                                                let _e8182_ = phi_32045_;
                                                let _e8183_ = select(_e8163_, 1f, _e8182_);
                                                if (_e8168_ != _e8168_) {
                                                    phi_32060_ = true;
                                                } else {
                                                    phi_32060_ = (1f <= _e8168_);
                                                }
                                                let _e8187_ = phi_32060_;
                                                let _e8188_ = select(_e8168_, 1f, _e8187_);
                                                if (_e8173_ != _e8173_) {
                                                    phi_32075_ = true;
                                                } else {
                                                    phi_32075_ = (1f <= _e8173_);
                                                }
                                                let _e8192_ = phi_32075_;
                                                let _e8193_ = select(_e8173_, 1f, _e8192_);
                                                if (_e8141_ != _e8141_) {
                                                    phi_32090_ = true;
                                                } else {
                                                    phi_32090_ = (-1f >= _e8141_);
                                                }
                                                let _e8197_ = phi_32090_;
                                                let _e8198_ = select(_e8141_, -1f, _e8197_);
                                                if (_e8145_ != _e8145_) {
                                                    phi_32105_ = true;
                                                } else {
                                                    phi_32105_ = (-1f >= _e8145_);
                                                }
                                                let _e8202_ = phi_32105_;
                                                let _e8203_ = select(_e8145_, -1f, _e8202_);
                                                if (_e8149_ != _e8149_) {
                                                    phi_32120_ = true;
                                                } else {
                                                    phi_32120_ = (-1f >= _e8149_);
                                                }
                                                let _e8207_ = phi_32120_;
                                                let _e8208_ = select(_e8149_, -1f, _e8207_);
                                                if (_e8153_ != _e8153_) {
                                                    phi_32135_ = true;
                                                } else {
                                                    phi_32135_ = (-1f >= _e8153_);
                                                }
                                                let _e8212_ = phi_32135_;
                                                let _e8213_ = select(_e8153_, -1f, _e8212_);
                                                if (_e8198_ != _e8198_) {
                                                    phi_32150_ = true;
                                                } else {
                                                    phi_32150_ = (1f <= _e8198_);
                                                }
                                                let _e8217_ = phi_32150_;
                                                let _e8218_ = select(_e8198_, 1f, _e8217_);
                                                if (_e8203_ != _e8203_) {
                                                    phi_32165_ = true;
                                                } else {
                                                    phi_32165_ = (1f <= _e8203_);
                                                }
                                                let _e8222_ = phi_32165_;
                                                let _e8223_ = select(_e8203_, 1f, _e8222_);
                                                if (_e8208_ != _e8208_) {
                                                    phi_32180_ = true;
                                                } else {
                                                    phi_32180_ = (1f <= _e8208_);
                                                }
                                                let _e8227_ = phi_32180_;
                                                let _e8228_ = select(_e8208_, 1f, _e8227_);
                                                if (_e8213_ != _e8213_) {
                                                    phi_32195_ = true;
                                                } else {
                                                    phi_32195_ = (1f <= _e8213_);
                                                }
                                                let _e8232_ = phi_32195_;
                                                let _e8233_ = select(_e8213_, 1f, _e8232_);
                                                let _e8238_ = (0.5f * (_e8218_ + _e8178_));
                                                let _e8239_ = (0.5f * (_e8223_ + _e8183_));
                                                let _e8240_ = (0.5f * (_e8228_ + _e8188_));
                                                let _e8241_ = (0.5f * (_e8233_ + _e8193_));
                                                let _e8242_ = (_e8218_ - _e8178_);
                                                let _e8243_ = (_e8223_ - _e8183_);
                                                let _e8244_ = (_e8228_ - _e8188_);
                                                let _e8245_ = (_e8233_ - _e8193_);
                                                let _e8246_ = (0.5f * _e8242_);
                                                let _e8247_ = (0.5f * _e8243_);
                                                let _e8248_ = (0.5f * _e8244_);
                                                let _e8249_ = (0.5f * _e8245_);
                                                let _e8250_ = (_e8238_ * _e8238_);
                                                let _e8251_ = (_e8239_ * _e8239_);
                                                let _e8252_ = (_e8240_ * _e8240_);
                                                let _e8253_ = (_e8241_ * _e8241_);
                                                let _e8254_ = (_e8246_ * _e8246_);
                                                let _e8255_ = (_e8247_ * _e8247_);
                                                let _e8256_ = (_e8248_ * _e8248_);
                                                let _e8257_ = (_e8249_ * _e8249_);
                                                if (_e8242_ != _e8242_) {
                                                    phi_32210_ = true;
                                                } else {
                                                    phi_32210_ = (0f >= _e8242_);
                                                }
                                                let _e8261_ = phi_32210_;
                                                if (_e8243_ != _e8243_) {
                                                    phi_32225_ = true;
                                                } else {
                                                    phi_32225_ = (0f >= _e8243_);
                                                }
                                                let _e8266_ = phi_32225_;
                                                if (_e8244_ != _e8244_) {
                                                    phi_32240_ = true;
                                                } else {
                                                    phi_32240_ = (0f >= _e8244_);
                                                }
                                                let _e8271_ = phi_32240_;
                                                if (_e8245_ != _e8245_) {
                                                    phi_32255_ = true;
                                                } else {
                                                    phi_32255_ = (0f >= _e8245_);
                                                }
                                                let _e8276_ = phi_32255_;
                                                let _e8278_ = (_e8141_ - _e8122_);
                                                let _e8279_ = (_e8145_ - _e8127_);
                                                let _e8280_ = (_e8149_ - _e8132_);
                                                let _e8281_ = (_e8153_ - _e8137_);
                                                let _e8284_ = bitcast<f32>((bitcast<u32>(_e8066_) & 2147483647u));
                                                let _e8287_ = bitcast<f32>((bitcast<u32>(_e8071_) & 2147483647u));
                                                let _e8290_ = bitcast<f32>((bitcast<u32>(_e8076_) & 2147483647u));
                                                let _e8293_ = bitcast<f32>((bitcast<u32>(_e8081_) & 2147483647u));
                                                let _e8296_ = bitcast<f32>((bitcast<u32>(_e8102_) & 2147483647u));
                                                let _e8299_ = bitcast<f32>((bitcast<u32>(_e8107_) & 2147483647u));
                                                let _e8302_ = bitcast<f32>((bitcast<u32>(_e8112_) & 2147483647u));
                                                let _e8305_ = bitcast<f32>((bitcast<u32>(_e8117_) & 2147483647u));
                                                if (_e8284_ != _e8284_) {
                                                    phi_32510_ = true;
                                                } else {
                                                    phi_32510_ = (_e8296_ >= _e8284_);
                                                }
                                                let _e8309_ = phi_32510_;
                                                if (_e8287_ != _e8287_) {
                                                    phi_32525_ = true;
                                                } else {
                                                    phi_32525_ = (_e8299_ >= _e8287_);
                                                }
                                                let _e8314_ = phi_32525_;
                                                if (_e8290_ != _e8290_) {
                                                    phi_32540_ = true;
                                                } else {
                                                    phi_32540_ = (_e8302_ >= _e8290_);
                                                }
                                                let _e8319_ = phi_32540_;
                                                if (_e8293_ != _e8293_) {
                                                    phi_32555_ = true;
                                                } else {
                                                    phi_32555_ = (_e8305_ >= _e8293_);
                                                }
                                                let _e8324_ = phi_32555_;
                                                let _e8330_ = (1f - (_e8250_ + _e8254_));
                                                let _e8331_ = (1f - (_e8251_ + _e8255_));
                                                let _e8332_ = (1f - (_e8252_ + _e8256_));
                                                let _e8333_ = (1f - (_e8253_ + _e8257_));
                                                let _e8359_ = select(((_e8238_ * _e8330_) * ((_e8330_ * _e8330_) + ((4f * _e8250_) * _e8254_))), 0f, (select(_e8284_, _e8296_, _e8309_) <= 0f));
                                                let _e8361_ = select(((_e8239_ * _e8331_) * ((_e8331_ * _e8331_) + ((4f * _e8251_) * _e8255_))), 0f, (select(_e8287_, _e8299_, _e8314_) <= 0f));
                                                let _e8363_ = select(((_e8240_ * _e8332_) * ((_e8332_ * _e8332_) + ((4f * _e8252_) * _e8256_))), 0f, (select(_e8290_, _e8302_, _e8319_) <= 0f));
                                                let _e8365_ = select(((_e8241_ * _e8333_) * ((_e8333_ * _e8333_) + ((4f * _e8253_) * _e8257_))), 0f, (select(_e8293_, _e8305_, _e8324_) <= 0f));
                                                if (_e8278_ != _e8278_) {
                                                    phi_32570_ = true;
                                                } else {
                                                    phi_32570_ = (0.000001f >= _e8278_);
                                                }
                                                let _e8369_ = phi_32570_;
                                                let _e8370_ = select(_e8278_, 0.000001f, _e8369_);
                                                if (_e8279_ != _e8279_) {
                                                    phi_32585_ = true;
                                                } else {
                                                    phi_32585_ = (0.000001f >= _e8279_);
                                                }
                                                let _e8374_ = phi_32585_;
                                                let _e8375_ = select(_e8279_, 0.000001f, _e8374_);
                                                if (_e8280_ != _e8280_) {
                                                    phi_32600_ = true;
                                                } else {
                                                    phi_32600_ = (0.000001f >= _e8280_);
                                                }
                                                let _e8379_ = phi_32600_;
                                                let _e8380_ = select(_e8280_, 0.000001f, _e8379_);
                                                if (_e8281_ != _e8281_) {
                                                    phi_32615_ = true;
                                                } else {
                                                    phi_32615_ = (0.000001f >= _e8281_);
                                                }
                                                let _e8384_ = phi_32615_;
                                                let _e8385_ = select(_e8281_, 0.000001f, _e8384_);
                                                let _e8388_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8370_)));
                                                let _e8391_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8375_)));
                                                let _e8394_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8380_)));
                                                let _e8397_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8385_)));
                                                let _e8407_ = (_e8388_ + (_e8388_ * (1f - (_e8370_ * _e8388_))));
                                                let _e8409_ = (_e8391_ + (_e8391_ * (1f - (_e8375_ * _e8391_))));
                                                let _e8411_ = (_e8394_ + (_e8394_ * (1f - (_e8380_ * _e8394_))));
                                                let _e8413_ = (_e8397_ + (_e8397_ * (1f - (_e8385_ * _e8397_))));
                                                if _e7390_ {
                                                    phi_32630_ = true;
                                                } else {
                                                    phi_32630_ = (-1000000000000000000f >= _e7383_);
                                                }
                                                let _e8492_ = phi_32630_;
                                                let _e8493_ = select(_e7383_, -1000000000000000000f, _e8492_);
                                                if _e7395_ {
                                                    phi_32645_ = true;
                                                } else {
                                                    phi_32645_ = (-1000000000000000000f >= _e7385_);
                                                }
                                                let _e8496_ = phi_32645_;
                                                let _e8497_ = select(_e7385_, -1000000000000000000f, _e8496_);
                                                if _e7400_ {
                                                    phi_32660_ = true;
                                                } else {
                                                    phi_32660_ = (-1000000000000000000f >= _e7387_);
                                                }
                                                let _e8500_ = phi_32660_;
                                                let _e8501_ = select(_e7387_, -1000000000000000000f, _e8500_);
                                                if _e7405_ {
                                                    phi_32675_ = true;
                                                } else {
                                                    phi_32675_ = (-1000000000000000000f >= _e7389_);
                                                }
                                                let _e8504_ = phi_32675_;
                                                let _e8505_ = select(_e7389_, -1000000000000000000f, _e8504_);
                                                if (_e8493_ != _e8493_) {
                                                    phi_32690_ = true;
                                                } else {
                                                    phi_32690_ = (1000000000000000000f <= _e8493_);
                                                }
                                                let _e8509_ = phi_32690_;
                                                let _e8510_ = select(_e8493_, 1000000000000000000f, _e8509_);
                                                if (_e8497_ != _e8497_) {
                                                    phi_32705_ = true;
                                                } else {
                                                    phi_32705_ = (1000000000000000000f <= _e8497_);
                                                }
                                                let _e8514_ = phi_32705_;
                                                let _e8515_ = select(_e8497_, 1000000000000000000f, _e8514_);
                                                if (_e8501_ != _e8501_) {
                                                    phi_32720_ = true;
                                                } else {
                                                    phi_32720_ = (1000000000000000000f <= _e8501_);
                                                }
                                                let _e8519_ = phi_32720_;
                                                let _e8520_ = select(_e8501_, 1000000000000000000f, _e8519_);
                                                if (_e8505_ != _e8505_) {
                                                    phi_32735_ = true;
                                                } else {
                                                    phi_32735_ = (1000000000000000000f <= _e8505_);
                                                }
                                                let _e8524_ = phi_32735_;
                                                let _e8525_ = select(_e8505_, 1000000000000000000f, _e8524_);
                                                if _e7450_ {
                                                    phi_32750_ = true;
                                                } else {
                                                    phi_32750_ = (-1000000000000000000f >= _e7443_);
                                                }
                                                let _e8528_ = phi_32750_;
                                                let _e8529_ = select(_e7443_, -1000000000000000000f, _e8528_);
                                                if _e7455_ {
                                                    phi_32765_ = true;
                                                } else {
                                                    phi_32765_ = (-1000000000000000000f >= _e7445_);
                                                }
                                                let _e8532_ = phi_32765_;
                                                let _e8533_ = select(_e7445_, -1000000000000000000f, _e8532_);
                                                if _e7460_ {
                                                    phi_32780_ = true;
                                                } else {
                                                    phi_32780_ = (-1000000000000000000f >= _e7447_);
                                                }
                                                let _e8536_ = phi_32780_;
                                                let _e8537_ = select(_e7447_, -1000000000000000000f, _e8536_);
                                                if _e7465_ {
                                                    phi_32795_ = true;
                                                } else {
                                                    phi_32795_ = (-1000000000000000000f >= _e7449_);
                                                }
                                                let _e8540_ = phi_32795_;
                                                let _e8541_ = select(_e7449_, -1000000000000000000f, _e8540_);
                                                if (_e8529_ != _e8529_) {
                                                    phi_32810_ = true;
                                                } else {
                                                    phi_32810_ = (1000000000000000000f <= _e8529_);
                                                }
                                                let _e8545_ = phi_32810_;
                                                let _e8546_ = select(_e8529_, 1000000000000000000f, _e8545_);
                                                if (_e8533_ != _e8533_) {
                                                    phi_32825_ = true;
                                                } else {
                                                    phi_32825_ = (1000000000000000000f <= _e8533_);
                                                }
                                                let _e8550_ = phi_32825_;
                                                let _e8551_ = select(_e8533_, 1000000000000000000f, _e8550_);
                                                if (_e8537_ != _e8537_) {
                                                    phi_32840_ = true;
                                                } else {
                                                    phi_32840_ = (1000000000000000000f <= _e8537_);
                                                }
                                                let _e8555_ = phi_32840_;
                                                let _e8556_ = select(_e8537_, 1000000000000000000f, _e8555_);
                                                if (_e8541_ != _e8541_) {
                                                    phi_32855_ = true;
                                                } else {
                                                    phi_32855_ = (1000000000000000000f <= _e8541_);
                                                }
                                                let _e8560_ = phi_32855_;
                                                let _e8561_ = select(_e8541_, 1000000000000000000f, _e8560_);
                                                let _e8562_ = (_e8510_ != _e8510_);
                                                if _e8562_ {
                                                    phi_32870_ = true;
                                                } else {
                                                    phi_32870_ = (_e8546_ <= _e8510_);
                                                }
                                                let _e8565_ = phi_32870_;
                                                let _e8566_ = select(_e8510_, _e8546_, _e8565_);
                                                let _e8567_ = (_e8515_ != _e8515_);
                                                if _e8567_ {
                                                    phi_32885_ = true;
                                                } else {
                                                    phi_32885_ = (_e8551_ <= _e8515_);
                                                }
                                                let _e8570_ = phi_32885_;
                                                let _e8571_ = select(_e8515_, _e8551_, _e8570_);
                                                let _e8572_ = (_e8520_ != _e8520_);
                                                if _e8572_ {
                                                    phi_32900_ = true;
                                                } else {
                                                    phi_32900_ = (_e8556_ <= _e8520_);
                                                }
                                                let _e8575_ = phi_32900_;
                                                let _e8576_ = select(_e8520_, _e8556_, _e8575_);
                                                let _e8577_ = (_e8525_ != _e8525_);
                                                if _e8577_ {
                                                    phi_32915_ = true;
                                                } else {
                                                    phi_32915_ = (_e8561_ <= _e8525_);
                                                }
                                                let _e8580_ = phi_32915_;
                                                let _e8581_ = select(_e8525_, _e8561_, _e8580_);
                                                if _e8562_ {
                                                    phi_32930_ = true;
                                                } else {
                                                    phi_32930_ = (_e8546_ >= _e8510_);
                                                }
                                                let _e8584_ = phi_32930_;
                                                let _e8585_ = select(_e8510_, _e8546_, _e8584_);
                                                if _e8567_ {
                                                    phi_32945_ = true;
                                                } else {
                                                    phi_32945_ = (_e8551_ >= _e8515_);
                                                }
                                                let _e8588_ = phi_32945_;
                                                let _e8589_ = select(_e8515_, _e8551_, _e8588_);
                                                if _e8572_ {
                                                    phi_32960_ = true;
                                                } else {
                                                    phi_32960_ = (_e8556_ >= _e8520_);
                                                }
                                                let _e8592_ = phi_32960_;
                                                let _e8593_ = select(_e8520_, _e8556_, _e8592_);
                                                if _e8577_ {
                                                    phi_32975_ = true;
                                                } else {
                                                    phi_32975_ = (_e8561_ >= _e8525_);
                                                }
                                                let _e8596_ = phi_32975_;
                                                let _e8597_ = select(_e8525_, _e8561_, _e8596_);
                                                if (_e8566_ != _e8566_) {
                                                    phi_32990_ = true;
                                                } else {
                                                    phi_32990_ = (-1f >= _e8566_);
                                                }
                                                let _e8601_ = phi_32990_;
                                                let _e8602_ = select(_e8566_, -1f, _e8601_);
                                                if (_e8571_ != _e8571_) {
                                                    phi_33005_ = true;
                                                } else {
                                                    phi_33005_ = (-1f >= _e8571_);
                                                }
                                                let _e8606_ = phi_33005_;
                                                let _e8607_ = select(_e8571_, -1f, _e8606_);
                                                if (_e8576_ != _e8576_) {
                                                    phi_33020_ = true;
                                                } else {
                                                    phi_33020_ = (-1f >= _e8576_);
                                                }
                                                let _e8611_ = phi_33020_;
                                                let _e8612_ = select(_e8576_, -1f, _e8611_);
                                                if (_e8581_ != _e8581_) {
                                                    phi_33035_ = true;
                                                } else {
                                                    phi_33035_ = (-1f >= _e8581_);
                                                }
                                                let _e8616_ = phi_33035_;
                                                let _e8617_ = select(_e8581_, -1f, _e8616_);
                                                if (_e8602_ != _e8602_) {
                                                    phi_33050_ = true;
                                                } else {
                                                    phi_33050_ = (1f <= _e8602_);
                                                }
                                                let _e8621_ = phi_33050_;
                                                let _e8622_ = select(_e8602_, 1f, _e8621_);
                                                if (_e8607_ != _e8607_) {
                                                    phi_33065_ = true;
                                                } else {
                                                    phi_33065_ = (1f <= _e8607_);
                                                }
                                                let _e8626_ = phi_33065_;
                                                let _e8627_ = select(_e8607_, 1f, _e8626_);
                                                if (_e8612_ != _e8612_) {
                                                    phi_33080_ = true;
                                                } else {
                                                    phi_33080_ = (1f <= _e8612_);
                                                }
                                                let _e8631_ = phi_33080_;
                                                let _e8632_ = select(_e8612_, 1f, _e8631_);
                                                if (_e8617_ != _e8617_) {
                                                    phi_33095_ = true;
                                                } else {
                                                    phi_33095_ = (1f <= _e8617_);
                                                }
                                                let _e8636_ = phi_33095_;
                                                let _e8637_ = select(_e8617_, 1f, _e8636_);
                                                if (_e8585_ != _e8585_) {
                                                    phi_33110_ = true;
                                                } else {
                                                    phi_33110_ = (-1f >= _e8585_);
                                                }
                                                let _e8641_ = phi_33110_;
                                                let _e8642_ = select(_e8585_, -1f, _e8641_);
                                                if (_e8589_ != _e8589_) {
                                                    phi_33125_ = true;
                                                } else {
                                                    phi_33125_ = (-1f >= _e8589_);
                                                }
                                                let _e8646_ = phi_33125_;
                                                let _e8647_ = select(_e8589_, -1f, _e8646_);
                                                if (_e8593_ != _e8593_) {
                                                    phi_33140_ = true;
                                                } else {
                                                    phi_33140_ = (-1f >= _e8593_);
                                                }
                                                let _e8651_ = phi_33140_;
                                                let _e8652_ = select(_e8593_, -1f, _e8651_);
                                                if (_e8597_ != _e8597_) {
                                                    phi_33155_ = true;
                                                } else {
                                                    phi_33155_ = (-1f >= _e8597_);
                                                }
                                                let _e8656_ = phi_33155_;
                                                let _e8657_ = select(_e8597_, -1f, _e8656_);
                                                if (_e8642_ != _e8642_) {
                                                    phi_33170_ = true;
                                                } else {
                                                    phi_33170_ = (1f <= _e8642_);
                                                }
                                                let _e8661_ = phi_33170_;
                                                let _e8662_ = select(_e8642_, 1f, _e8661_);
                                                if (_e8647_ != _e8647_) {
                                                    phi_33185_ = true;
                                                } else {
                                                    phi_33185_ = (1f <= _e8647_);
                                                }
                                                let _e8666_ = phi_33185_;
                                                let _e8667_ = select(_e8647_, 1f, _e8666_);
                                                if (_e8652_ != _e8652_) {
                                                    phi_33200_ = true;
                                                } else {
                                                    phi_33200_ = (1f <= _e8652_);
                                                }
                                                let _e8671_ = phi_33200_;
                                                let _e8672_ = select(_e8652_, 1f, _e8671_);
                                                if (_e8657_ != _e8657_) {
                                                    phi_33215_ = true;
                                                } else {
                                                    phi_33215_ = (1f <= _e8657_);
                                                }
                                                let _e8676_ = phi_33215_;
                                                let _e8677_ = select(_e8657_, 1f, _e8676_);
                                                let _e8682_ = (0.5f * (_e8662_ + _e8622_));
                                                let _e8683_ = (0.5f * (_e8667_ + _e8627_));
                                                let _e8684_ = (0.5f * (_e8672_ + _e8632_));
                                                let _e8685_ = (0.5f * (_e8677_ + _e8637_));
                                                let _e8686_ = (_e8662_ - _e8622_);
                                                let _e8687_ = (_e8667_ - _e8627_);
                                                let _e8688_ = (_e8672_ - _e8632_);
                                                let _e8689_ = (_e8677_ - _e8637_);
                                                let _e8690_ = (0.5f * _e8686_);
                                                let _e8691_ = (0.5f * _e8687_);
                                                let _e8692_ = (0.5f * _e8688_);
                                                let _e8693_ = (0.5f * _e8689_);
                                                let _e8694_ = (_e8682_ * _e8682_);
                                                let _e8695_ = (_e8683_ * _e8683_);
                                                let _e8696_ = (_e8684_ * _e8684_);
                                                let _e8697_ = (_e8685_ * _e8685_);
                                                let _e8698_ = (_e8690_ * _e8690_);
                                                let _e8699_ = (_e8691_ * _e8691_);
                                                let _e8700_ = (_e8692_ * _e8692_);
                                                let _e8701_ = (_e8693_ * _e8693_);
                                                if (_e8686_ != _e8686_) {
                                                    phi_33230_ = true;
                                                } else {
                                                    phi_33230_ = (0f >= _e8686_);
                                                }
                                                let _e8705_ = phi_33230_;
                                                if (_e8687_ != _e8687_) {
                                                    phi_33245_ = true;
                                                } else {
                                                    phi_33245_ = (0f >= _e8687_);
                                                }
                                                let _e8710_ = phi_33245_;
                                                if (_e8688_ != _e8688_) {
                                                    phi_33260_ = true;
                                                } else {
                                                    phi_33260_ = (0f >= _e8688_);
                                                }
                                                let _e8715_ = phi_33260_;
                                                if (_e8689_ != _e8689_) {
                                                    phi_33275_ = true;
                                                } else {
                                                    phi_33275_ = (0f >= _e8689_);
                                                }
                                                let _e8720_ = phi_33275_;
                                                let _e8722_ = (_e8585_ - _e8566_);
                                                let _e8723_ = (_e8589_ - _e8571_);
                                                let _e8724_ = (_e8593_ - _e8576_);
                                                let _e8725_ = (_e8597_ - _e8581_);
                                                let _e8728_ = bitcast<f32>((bitcast<u32>(_e8510_) & 2147483647u));
                                                let _e8731_ = bitcast<f32>((bitcast<u32>(_e8515_) & 2147483647u));
                                                let _e8734_ = bitcast<f32>((bitcast<u32>(_e8520_) & 2147483647u));
                                                let _e8737_ = bitcast<f32>((bitcast<u32>(_e8525_) & 2147483647u));
                                                let _e8740_ = bitcast<f32>((bitcast<u32>(_e8546_) & 2147483647u));
                                                let _e8743_ = bitcast<f32>((bitcast<u32>(_e8551_) & 2147483647u));
                                                let _e8746_ = bitcast<f32>((bitcast<u32>(_e8556_) & 2147483647u));
                                                let _e8749_ = bitcast<f32>((bitcast<u32>(_e8561_) & 2147483647u));
                                                if (_e8728_ != _e8728_) {
                                                    phi_33530_ = true;
                                                } else {
                                                    phi_33530_ = (_e8740_ >= _e8728_);
                                                }
                                                let _e8753_ = phi_33530_;
                                                if (_e8731_ != _e8731_) {
                                                    phi_33545_ = true;
                                                } else {
                                                    phi_33545_ = (_e8743_ >= _e8731_);
                                                }
                                                let _e8758_ = phi_33545_;
                                                if (_e8734_ != _e8734_) {
                                                    phi_33560_ = true;
                                                } else {
                                                    phi_33560_ = (_e8746_ >= _e8734_);
                                                }
                                                let _e8763_ = phi_33560_;
                                                if (_e8737_ != _e8737_) {
                                                    phi_33575_ = true;
                                                } else {
                                                    phi_33575_ = (_e8749_ >= _e8737_);
                                                }
                                                let _e8768_ = phi_33575_;
                                                let _e8774_ = (1f - (_e8694_ + _e8698_));
                                                let _e8775_ = (1f - (_e8695_ + _e8699_));
                                                let _e8776_ = (1f - (_e8696_ + _e8700_));
                                                let _e8777_ = (1f - (_e8697_ + _e8701_));
                                                let _e8803_ = select(((_e8682_ * _e8774_) * ((_e8774_ * _e8774_) + ((4f * _e8694_) * _e8698_))), 0f, (select(_e8728_, _e8740_, _e8753_) <= 0f));
                                                let _e8805_ = select(((_e8683_ * _e8775_) * ((_e8775_ * _e8775_) + ((4f * _e8695_) * _e8699_))), 0f, (select(_e8731_, _e8743_, _e8758_) <= 0f));
                                                let _e8807_ = select(((_e8684_ * _e8776_) * ((_e8776_ * _e8776_) + ((4f * _e8696_) * _e8700_))), 0f, (select(_e8734_, _e8746_, _e8763_) <= 0f));
                                                let _e8809_ = select(((_e8685_ * _e8777_) * ((_e8777_ * _e8777_) + ((4f * _e8697_) * _e8701_))), 0f, (select(_e8737_, _e8749_, _e8768_) <= 0f));
                                                if (_e8722_ != _e8722_) {
                                                    phi_33590_ = true;
                                                } else {
                                                    phi_33590_ = (0.000001f >= _e8722_);
                                                }
                                                let _e8813_ = phi_33590_;
                                                let _e8814_ = select(_e8722_, 0.000001f, _e8813_);
                                                if (_e8723_ != _e8723_) {
                                                    phi_33605_ = true;
                                                } else {
                                                    phi_33605_ = (0.000001f >= _e8723_);
                                                }
                                                let _e8818_ = phi_33605_;
                                                let _e8819_ = select(_e8723_, 0.000001f, _e8818_);
                                                if (_e8724_ != _e8724_) {
                                                    phi_33620_ = true;
                                                } else {
                                                    phi_33620_ = (0.000001f >= _e8724_);
                                                }
                                                let _e8823_ = phi_33620_;
                                                let _e8824_ = select(_e8724_, 0.000001f, _e8823_);
                                                if (_e8725_ != _e8725_) {
                                                    phi_33635_ = true;
                                                } else {
                                                    phi_33635_ = (0.000001f >= _e8725_);
                                                }
                                                let _e8828_ = phi_33635_;
                                                let _e8829_ = select(_e8725_, 0.000001f, _e8828_);
                                                let _e8832_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8814_)));
                                                let _e8835_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8819_)));
                                                let _e8838_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8824_)));
                                                let _e8841_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8829_)));
                                                let _e8851_ = (_e8832_ + (_e8832_ * (1f - (_e8814_ * _e8832_))));
                                                let _e8853_ = (_e8835_ + (_e8835_ * (1f - (_e8819_ * _e8835_))));
                                                let _e8855_ = (_e8838_ + (_e8838_ * (1f - (_e8824_ * _e8838_))));
                                                let _e8857_ = (_e8841_ + (_e8841_ * (1f - (_e8829_ * _e8841_))));
                                                let _e8934_ = (1f - _e489_);
                                                let _e8935_ = (1f - _e509_);
                                                let _e8936_ = (1f - _e529_);
                                                let _e8937_ = (1f - _e549_);
                                                phi_22595_ = type_30_(array<f32, 4>(((_e8028_.member.member[0] * _e8934_) + (_e489_ * select(select(select(_e8803_, ((select(_e8686_, 0f, _e8705_) * _e8803_) * (_e8851_ + (_e8851_ * (1f - (_e8814_ * _e8851_))))), (select((select(0f, 1f, (-1f <= _e8566_)) * select(0f, 1f, (_e8585_ <= 1f))), 1f, (_e8722_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8566_)), 0f, (_e8585_ <= -1f)))), ((_e8028_.member.member[1] * _e8935_) + (_e509_ * select(select(select(_e8805_, ((select(_e8687_, 0f, _e8710_) * _e8805_) * (_e8853_ + (_e8853_ * (1f - (_e8819_ * _e8853_))))), (select((select(0f, 1f, (-1f <= _e8571_)) * select(0f, 1f, (_e8589_ <= 1f))), 1f, (_e8723_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8571_)), 0f, (_e8589_ <= -1f)))), ((_e8028_.member.member[2] * _e8936_) + (_e529_ * select(select(select(_e8807_, ((select(_e8688_, 0f, _e8715_) * _e8807_) * (_e8855_ + (_e8855_ * (1f - (_e8824_ * _e8855_))))), (select((select(0f, 1f, (-1f <= _e8576_)) * select(0f, 1f, (_e8593_ <= 1f))), 1f, (_e8724_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8576_)), 0f, (_e8593_ <= -1f)))), ((_e8028_.member.member[3] * _e8937_) + (_e549_ * select(select(select(_e8809_, ((select(_e8689_, 0f, _e8720_) * _e8809_) * (_e8857_ + (_e8857_ * (1f - (_e8829_ * _e8857_))))), (select((select(0f, 1f, (-1f <= _e8581_)) * select(0f, 1f, (_e8597_ <= 1f))), 1f, (_e8725_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8581_)), 0f, (_e8597_ <= -1f))))));
                                                phi_22596_ = type_30_(array<f32, 4>(((_e7368_.member.member[0] * _e8934_) + (_e489_ * select(select(select(_e8359_, ((select(_e8242_, 0f, _e8261_) * _e8359_) * (_e8407_ + (_e8407_ * (1f - (_e8370_ * _e8407_))))), (select((select(0f, 1f, (-1f <= _e8122_)) * select(0f, 1f, (_e8141_ <= 1f))), 1f, (_e8278_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8122_)), 0f, (_e8141_ <= -1f)))), ((_e7368_.member.member[1] * _e8935_) + (_e509_ * select(select(select(_e8361_, ((select(_e8243_, 0f, _e8266_) * _e8361_) * (_e8409_ + (_e8409_ * (1f - (_e8375_ * _e8409_))))), (select((select(0f, 1f, (-1f <= _e8127_)) * select(0f, 1f, (_e8145_ <= 1f))), 1f, (_e8279_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8127_)), 0f, (_e8145_ <= -1f)))), ((_e7368_.member.member[2] * _e8936_) + (_e529_ * select(select(select(_e8363_, ((select(_e8244_, 0f, _e8271_) * _e8363_) * (_e8411_ + (_e8411_ * (1f - (_e8380_ * _e8411_))))), (select((select(0f, 1f, (-1f <= _e8132_)) * select(0f, 1f, (_e8149_ <= 1f))), 1f, (_e8280_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8132_)), 0f, (_e8149_ <= -1f)))), ((_e7368_.member.member[3] * _e8937_) + (_e549_ * select(select(select(_e8365_, ((select(_e8245_, 0f, _e8276_) * _e8365_) * (_e8413_ + (_e8413_ * (1f - (_e8385_ * _e8413_))))), (select((select(0f, 1f, (-1f <= _e8137_)) * select(0f, 1f, (_e8153_ <= 1f))), 1f, (_e8281_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8137_)), 0f, (_e8153_ <= -1f))))));
                                            }
                                            let _e8983_ = phi_22595_;
                                            let _e8985_ = phi_22596_;
                                            let _e9046_ = (_e1586_ * 6u);
                                            let _e9047_ = local_9_;
                                            let _e9052_ = bitcast<u32>(_e9047_[0]);
                                            let _e9053_ = (_e9052_ & 2139095040u);
                                            if (_e9053_ == 2139095040u) {
                                                phi_22752_ = true;
                                            } else {
                                                phi_22752_ = (_e9053_ == 0u);
                                            }
                                            let _e9057_ = phi_22752_;
                                            let _e9060_ = bitcast<u32>(_e9047_[1]);
                                            let _e9061_ = (_e9060_ & 2139095040u);
                                            if (_e9061_ == 2139095040u) {
                                                phi_22768_ = true;
                                            } else {
                                                phi_22768_ = (_e9061_ == 0u);
                                            }
                                            let _e9065_ = phi_22768_;
                                            let _e9068_ = bitcast<u32>(_e9047_[2]);
                                            let _e9069_ = (_e9068_ & 2139095040u);
                                            if (_e9069_ == 2139095040u) {
                                                phi_22784_ = true;
                                            } else {
                                                phi_22784_ = (_e9069_ == 0u);
                                            }
                                            let _e9073_ = phi_22784_;
                                            let _e9076_ = bitcast<u32>(_e9047_[3]);
                                            let _e9077_ = (_e9076_ & 2139095040u);
                                            if (_e9077_ == 2139095040u) {
                                                phi_22800_ = true;
                                            } else {
                                                phi_22800_ = (_e9077_ == 0u);
                                            }
                                            let _e9081_ = phi_22800_;
                                            let _e9085_ = (_e9046_ < _e162_);
                                            if _e9085_ {
                                                if _e9085_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9046_] = vec4<f32>(bitcast<f32>(select(_e9052_, 0u, _e9057_)), bitcast<f32>(select(_e9060_, 0u, _e9065_)), bitcast<f32>(select(_e9068_, 0u, _e9073_)), bitcast<f32>(select(_e9076_, 0u, _e9081_)));
                                            }
                                            let _e9088_ = (_e9046_ + 1u);
                                            let _e9089_ = local_10_;
                                            let _e9094_ = bitcast<u32>(_e9089_[0]);
                                            let _e9095_ = (_e9094_ & 2139095040u);
                                            if (_e9095_ == 2139095040u) {
                                                phi_22838_ = true;
                                            } else {
                                                phi_22838_ = (_e9095_ == 0u);
                                            }
                                            let _e9099_ = phi_22838_;
                                            let _e9102_ = bitcast<u32>(_e9089_[1]);
                                            let _e9103_ = (_e9102_ & 2139095040u);
                                            if (_e9103_ == 2139095040u) {
                                                phi_22854_ = true;
                                            } else {
                                                phi_22854_ = (_e9103_ == 0u);
                                            }
                                            let _e9107_ = phi_22854_;
                                            let _e9110_ = bitcast<u32>(_e9089_[2]);
                                            let _e9111_ = (_e9110_ & 2139095040u);
                                            if (_e9111_ == 2139095040u) {
                                                phi_22870_ = true;
                                            } else {
                                                phi_22870_ = (_e9111_ == 0u);
                                            }
                                            let _e9115_ = phi_22870_;
                                            let _e9118_ = bitcast<u32>(_e9089_[3]);
                                            let _e9119_ = (_e9118_ & 2139095040u);
                                            if (_e9119_ == 2139095040u) {
                                                phi_22886_ = true;
                                            } else {
                                                phi_22886_ = (_e9119_ == 0u);
                                            }
                                            let _e9123_ = phi_22886_;
                                            let _e9127_ = (_e9088_ < _e162_);
                                            if _e9127_ {
                                                if _e9127_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9088_] = vec4<f32>(bitcast<f32>(select(_e9094_, 0u, _e9099_)), bitcast<f32>(select(_e9102_, 0u, _e9107_)), bitcast<f32>(select(_e9110_, 0u, _e9115_)), bitcast<f32>(select(_e9118_, 0u, _e9123_)));
                                            }
                                            let _e9130_ = (_e9046_ + 2u);
                                            let _e9135_ = bitcast<u32>(_e8985_.member[0]);
                                            let _e9136_ = (_e9135_ & 2139095040u);
                                            if (_e9136_ == 2139095040u) {
                                                phi_22918_ = true;
                                            } else {
                                                phi_22918_ = (_e9136_ == 0u);
                                            }
                                            let _e9140_ = phi_22918_;
                                            let _e9143_ = bitcast<u32>(_e8985_.member[1]);
                                            let _e9144_ = (_e9143_ & 2139095040u);
                                            if (_e9144_ == 2139095040u) {
                                                phi_22934_ = true;
                                            } else {
                                                phi_22934_ = (_e9144_ == 0u);
                                            }
                                            let _e9148_ = phi_22934_;
                                            let _e9151_ = bitcast<u32>(_e8985_.member[2]);
                                            let _e9152_ = (_e9151_ & 2139095040u);
                                            if (_e9152_ == 2139095040u) {
                                                phi_22950_ = true;
                                            } else {
                                                phi_22950_ = (_e9152_ == 0u);
                                            }
                                            let _e9156_ = phi_22950_;
                                            let _e9159_ = bitcast<u32>(_e8985_.member[3]);
                                            let _e9160_ = (_e9159_ & 2139095040u);
                                            if (_e9160_ == 2139095040u) {
                                                phi_22966_ = true;
                                            } else {
                                                phi_22966_ = (_e9160_ == 0u);
                                            }
                                            let _e9164_ = phi_22966_;
                                            let _e9168_ = (_e9130_ < _e162_);
                                            if _e9168_ {
                                                if _e9168_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9130_] = vec4<f32>(bitcast<f32>(select(_e9135_, 0u, _e9140_)), bitcast<f32>(select(_e9143_, 0u, _e9148_)), bitcast<f32>(select(_e9151_, 0u, _e9156_)), bitcast<f32>(select(_e9159_, 0u, _e9164_)));
                                            }
                                            let _e9171_ = (_e9046_ + 3u);
                                            let _e9176_ = bitcast<u32>(_e8983_.member[0]);
                                            let _e9177_ = (_e9176_ & 2139095040u);
                                            if (_e9177_ == 2139095040u) {
                                                phi_22998_ = true;
                                            } else {
                                                phi_22998_ = (_e9177_ == 0u);
                                            }
                                            let _e9181_ = phi_22998_;
                                            let _e9184_ = bitcast<u32>(_e8983_.member[1]);
                                            let _e9185_ = (_e9184_ & 2139095040u);
                                            if (_e9185_ == 2139095040u) {
                                                phi_23014_ = true;
                                            } else {
                                                phi_23014_ = (_e9185_ == 0u);
                                            }
                                            let _e9189_ = phi_23014_;
                                            let _e9192_ = bitcast<u32>(_e8983_.member[2]);
                                            let _e9193_ = (_e9192_ & 2139095040u);
                                            if (_e9193_ == 2139095040u) {
                                                phi_23030_ = true;
                                            } else {
                                                phi_23030_ = (_e9193_ == 0u);
                                            }
                                            let _e9197_ = phi_23030_;
                                            let _e9200_ = bitcast<u32>(_e8983_.member[3]);
                                            let _e9201_ = (_e9200_ & 2139095040u);
                                            if (_e9201_ == 2139095040u) {
                                                phi_23046_ = true;
                                            } else {
                                                phi_23046_ = (_e9201_ == 0u);
                                            }
                                            let _e9205_ = phi_23046_;
                                            let _e9209_ = (_e9171_ < _e162_);
                                            if _e9209_ {
                                                if _e9209_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9171_] = vec4<f32>(bitcast<f32>(select(_e9176_, 0u, _e9181_)), bitcast<f32>(select(_e9184_, 0u, _e9189_)), bitcast<f32>(select(_e9192_, 0u, _e9197_)), bitcast<f32>(select(_e9200_, 0u, _e9205_)));
                                            }
                                            let _e9212_ = (_e9046_ + 4u);
                                            let _e9213_ = bitcast<u32>((_e7368_.member_1_.member[0] + (_e460_.x * (_e8985_.member[0] * _e8985_.member[1]))));
                                            let _e9214_ = (_e9213_ & 2139095040u);
                                            if (_e9214_ == 2139095040u) {
                                                phi_23078_ = true;
                                            } else {
                                                phi_23078_ = (_e9214_ == 0u);
                                            }
                                            let _e9218_ = phi_23078_;
                                            let _e9221_ = bitcast<u32>((_e7368_.member_1_.member[1] + (_e460_.y * (_e8985_.member[1] * _e8985_.member[2]))));
                                            let _e9222_ = (_e9221_ & 2139095040u);
                                            if (_e9222_ == 2139095040u) {
                                                phi_23094_ = true;
                                            } else {
                                                phi_23094_ = (_e9222_ == 0u);
                                            }
                                            let _e9226_ = phi_23094_;
                                            let _e9229_ = bitcast<u32>((_e7368_.member_1_.member[2] + (_e460_.z * (_e8985_.member[2] * _e8985_.member[3]))));
                                            let _e9230_ = (_e9229_ & 2139095040u);
                                            if (_e9230_ == 2139095040u) {
                                                phi_23110_ = true;
                                            } else {
                                                phi_23110_ = (_e9230_ == 0u);
                                            }
                                            let _e9234_ = phi_23110_;
                                            let _e9237_ = bitcast<u32>((_e7368_.member_1_.member[3] + (_e460_.w * (_e8985_.member[3] * _e8985_.member[0]))));
                                            let _e9238_ = (_e9237_ & 2139095040u);
                                            if (_e9238_ == 2139095040u) {
                                                phi_23126_ = true;
                                            } else {
                                                phi_23126_ = (_e9238_ == 0u);
                                            }
                                            let _e9242_ = phi_23126_;
                                            let _e9246_ = (_e9212_ < _e162_);
                                            if _e9246_ {
                                                if _e9246_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9212_] = vec4<f32>(bitcast<f32>(select(_e9213_, 0u, _e9218_)), bitcast<f32>(select(_e9221_, 0u, _e9226_)), bitcast<f32>(select(_e9229_, 0u, _e9234_)), bitcast<f32>(select(_e9237_, 0u, _e9242_)));
                                            }
                                            let _e9249_ = (_e9046_ + 5u);
                                            let _e9250_ = bitcast<u32>((_e8028_.member_1_.member[0] + (_e460_.x * (_e8983_.member[0] * _e8983_.member[1]))));
                                            let _e9251_ = (_e9250_ & 2139095040u);
                                            if (_e9251_ == 2139095040u) {
                                                phi_23158_ = true;
                                            } else {
                                                phi_23158_ = (_e9251_ == 0u);
                                            }
                                            let _e9255_ = phi_23158_;
                                            let _e9258_ = bitcast<u32>((_e8028_.member_1_.member[1] + (_e460_.y * (_e8983_.member[1] * _e8983_.member[2]))));
                                            let _e9259_ = (_e9258_ & 2139095040u);
                                            if (_e9259_ == 2139095040u) {
                                                phi_23174_ = true;
                                            } else {
                                                phi_23174_ = (_e9259_ == 0u);
                                            }
                                            let _e9263_ = phi_23174_;
                                            let _e9266_ = bitcast<u32>((_e8028_.member_1_.member[2] + (_e460_.z * (_e8983_.member[2] * _e8983_.member[3]))));
                                            let _e9267_ = (_e9266_ & 2139095040u);
                                            if (_e9267_ == 2139095040u) {
                                                phi_23190_ = true;
                                            } else {
                                                phi_23190_ = (_e9267_ == 0u);
                                            }
                                            let _e9271_ = phi_23190_;
                                            let _e9274_ = bitcast<u32>((_e8028_.member_1_.member[3] + (_e460_.w * (_e8983_.member[3] * _e8983_.member[0]))));
                                            let _e9275_ = (_e9274_ & 2139095040u);
                                            if (_e9275_ == 2139095040u) {
                                                phi_23206_ = true;
                                            } else {
                                                phi_23206_ = (_e9275_ == 0u);
                                            }
                                            let _e9279_ = phi_23206_;
                                            let _e9283_ = (_e9249_ < _e162_);
                                            if _e9283_ {
                                                if _e9283_ {
                                                } else {
                                                    phi_36246_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9249_] = vec4<f32>(bitcast<f32>(select(_e9250_, 0u, _e9255_)), bitcast<f32>(select(_e9258_, 0u, _e9263_)), bitcast<f32>(select(_e9266_, 0u, _e9271_)), bitcast<f32>(select(_e9274_, 0u, _e9279_)));
                                            }
                                            if ((_e164_ + 1u) == _e1581_) {
                                                phi_23230_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e9289_ = phi_23230_;
                                                    if (_e9289_.member < _e9289_.member_1_) {
                                                        phi_23231_ = type_15_((_e9289_.member + 1u), _e9289_.member_1_);
                                                        phi_23254_ = type_15_(1u, _e9289_.member);
                                                    } else {
                                                        phi_23231_ = _e9289_;
                                                        phi_23254_ = type_15_(0u, u32());
                                                    }
                                                    let _e9302_ = phi_23231_;
                                                    let _e9304_ = phi_23254_;
                                                    let _e9308_ = (bitcast<i32>(_e9304_.member) != 0i);
                                                    if _e9308_ {
                                                        let _e9309_ = (_e9304_.member_1_ < 4u);
                                                        if _e9309_ {
                                                        } else {
                                                            phi_35412_ = true;
                                                            break;
                                                        }
                                                        let _e9311_ = local_11_[_e9304_.member_1_];
                                                        let _e9312_ = bitcast<u32>(_e9311_);
                                                        let _e9313_ = (_e9312_ & 2139095040u);
                                                        if (_e9313_ == 2139095040u) {
                                                            phi_23279_ = true;
                                                        } else {
                                                            phi_23279_ = (_e9313_ == 0u);
                                                        }
                                                        let _e9317_ = phi_23279_;
                                                        let _e9319_ = bitcast<f32>(select(_e9312_, 0u, _e9317_));
                                                        if (_e9319_ < -1000000000000000000f) {
                                                            phi_23296_ = -1000000000000000000f;
                                                        } else {
                                                            phi_23296_ = select(_e9319_, 1000000000000000000f, (_e9319_ > 1000000000000000000f));
                                                        }
                                                        let _e9324_ = phi_23296_;
                                                        let _e9325_ = (24u + _e9304_.member_1_);
                                                        if (_e9325_ < 256u) {
                                                        } else {
                                                            phi_35412_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e9325_] = _e9324_;
                                                        if _e9309_ {
                                                        } else {
                                                            phi_35412_ = true;
                                                            break;
                                                        }
                                                        let _e9329_ = local_12_[_e9304_.member_1_];
                                                        let _e9330_ = bitcast<u32>(_e9329_);
                                                        let _e9331_ = (_e9330_ & 2139095040u);
                                                        if (_e9331_ == 2139095040u) {
                                                            phi_23321_ = true;
                                                        } else {
                                                            phi_23321_ = (_e9331_ == 0u);
                                                        }
                                                        let _e9335_ = phi_23321_;
                                                        let _e9337_ = bitcast<f32>(select(_e9330_, 0u, _e9335_));
                                                        if (_e9337_ < -1000000000000000000f) {
                                                            phi_23338_ = -1000000000000000000f;
                                                        } else {
                                                            phi_23338_ = select(_e9337_, 1000000000000000000f, (_e9337_ > 1000000000000000000f));
                                                        }
                                                        let _e9342_ = phi_23338_;
                                                        let _e9343_ = (28u + _e9304_.member_1_);
                                                        if (_e9343_ < 256u) {
                                                        } else {
                                                            phi_35412_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e9343_] = _e9342_;
                                                    }
                                                    phi_23230_ = _e9302_;
                                                    phi_35412_ = _e6689_;
                                                    if !(_e9308_) {
                                                        break;
                                                    }
                                                }
                                                let _e9348_ = phi_35412_;
                                                phi_36246_ = _e9348_;
                                                if _e9348_ {
                                                    break;
                                                }
                                                phi_36251_ = _e9348_;
                                            } else {
                                                phi_36251_ = _e6689_;
                                            }
                                            let _e9350_ = phi_36251_;
                                            phi_36250_ = _e9350_;
                                        } else {
                                            phi_36250_ = _e6612_;
                                        }
                                        let _e9352_ = phi_36250_;
                                        phi_36249_ = _e9352_;
                                    } else {
                                        phi_36249_ = _e6612_;
                                    }
                                    let _e9354_ = phi_36249_;
                                    workgroupBarrier();
                                    phi_36248_ = _e9354_;
                                } else {
                                    phi_36248_ = _e1552_;
                                }
                                let _e9356_ = phi_36248_;
                                phi_36247_ = _e9356_;
                            } else {
                                phi_36247_ = _e1552_;
                            }
                            let _e9358_ = phi_36247_;
                            phi_33863_ = _e9358_;
                            phi_2851_ = _e1567_;
                            phi_36246_ = _e9358_;
                            if !(_e1573_) {
                                break;
                            }
                        }
                        let _e9361_ = phi_36246_;
                        phi_36252_ = _e9361_;
                        if _e9361_ {
                            break;
                        }
                        phi_36253_ = _e9361_;
                    } else {
                        phi_36253_ = _e1161_;
                    }
                    let _e9363_ = phi_36253_;
                    phi_33866_ = _e9363_;
                    phi_2126_ = _e1176_;
                    phi_36252_ = _e9363_;
                    if !(_e1182_) {
                        break;
                    }
                }
                let _e9366_ = phi_36252_;
                if _e9366_ {
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
