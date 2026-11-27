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
    member: type_30_,
    member_1_: type_30_,
}

struct type_32_ {
    member: f32,
    member_1_: f32,
    member_2_: f32,
    member_3_: f32,
    member_4_: f32,
    member_5_: f32,
}

struct type_34_ {
    member: i32,
    member_1_: i32,
}

struct type_35_ {
    member: u32,
    member_1_: i32,
}

struct type_38_ {
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
    var phi_33936_: bool;
    var phi_34132_: bool;
    var phi_34131_: bool;
    var phi_2046_: bool;
    var phi_34130_: bool;
    var phi_2048_: bool;
    var phi_2052_: bool;
    var phi_2053_: bool;
    var phi_34129_: bool;
    var phi_2054_: bool;
    var phi_2057_: bool;
    var phi_2058_: bool;
    var phi_2059_: bool;
    var phi_2069_: bool;
    var phi_2076_: type_15_;
    var phi_2077_: type_15_;
    var phi_2100_: type_15_;
    var phi_34124_: bool;
    var phi_34165_: bool;
    var phi_34162_: bool;
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
    var phi_34159_: bool;
    var phi_2851_: type_15_;
    var phi_2852_: type_15_;
    var phi_2875_: type_15_;
    var phi_2890_: u32;
    var phi_2919_: vec4<f32>;
    var phi_2932_: vec4<f32>;
    var phi_2933_: vec4<f32>;
    var phi_2934_: vec4<f32>;
    var phi_34154_: bool;
    var phi_6752_: type_15_;
    var phi_6753_: type_15_;
    var phi_6776_: type_15_;
    var phi_6871_: f32;
    var phi_6882_: f32;
    var phi_6893_: f32;
    var phi_6904_: f32;
    var phi_6915_: f32;
    var phi_6926_: f32;
    var phi_6937_: f32;
    var phi_6948_: f32;
    var phi_6959_: f32;
    var phi_6970_: f32;
    var phi_7278_: f32;
    var phi_7289_: bool;
    var phi_7295_: f32;
    var phi_7300_: u32;
    var phi_7306_: u32;
    var phi_7339_: f32;
    var phi_7344_: f32;
    var phi_7361_: f32;
    var phi_7362_: f32;
    var phi_7363_: f32;
    var phi_7364_: f32;
    var phi_7381_: f32;
    var phi_7384_: f32;
    var phi_7386_: f32;
    var phi_7388_: f32;
    var phi_7390_: u32;
    var phi_7392_: f32;
    var phi_7394_: f32;
    var phi_7433_: f32;
    var phi_7434_: f32;
    var phi_7382_: f32;
    var phi_7385_: f32;
    var phi_7387_: f32;
    var phi_7389_: f32;
    var phi_7391_: u32;
    var phi_7393_: f32;
    var phi_7395_: f32;
    var phi_34133_: bool;
    var phi_7486_: f32;
    var phi_7487_: f32;
    var phi_7497_: u32;
    var phi_7516_: array<f32, 2>;
    var phi_7771_: bool;
    var phi_7794_: bool;
    var phi_7822_: bool;
    var phi_7846_: bool;
    var phi_34250_: bool;
    var phi_34249_: bool;
    var phi_5752_: type_15_;
    var phi_5753_: type_15_;
    var phi_5776_: type_15_;
    var phi_5794_: f32;
    var phi_5805_: bool;
    var phi_5830_: f32;
    var phi_5841_: bool;
    var phi_34266_: bool;
    var phi_34293_: bool;
    var phi_5969_: f32;
    var phi_5986_: f32;
    var phi_5997_: f32;
    var phi_6008_: f32;
    var phi_6019_: f32;
    var phi_6030_: f32;
    var phi_6245_: f32;
    var phi_6320_: f32;
    var phi_6325_: u32;
    var phi_6331_: u32;
    var phi_6356_: type_15_;
    var phi_6359_: f32;
    var phi_6361_: f32;
    var phi_6357_: type_15_;
    var phi_6384_: type_15_;
    var phi_23716_: bool;
    var phi_23776_: bool;
    var phi_23836_: bool;
    var phi_23896_: bool;
    var phi_6550_: bool;
    var phi_6604_: bool;
    var phi_6360_: f32;
    var phi_6362_: f32;
    var phi_34278_: bool;
    var local_22_: f32;
    var phi_6716_: bool;
    var local_23_: f32;
    var phi_6738_: bool;
    var phi_34588_: bool;
    var phi_34356_: bool;
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
    var phi_34344_: bool;
    var phi_3494_: type_15_;
    var phi_3497_: array<f32, 2>;
    var phi_3499_: f32;
    var phi_3501_: f32;
    var phi_3503_: array<f32, 2>;
    var phi_3505_: bool;
    var phi_3495_: type_15_;
    var phi_3528_: type_15_;
    var phi_23956_: bool;
    var phi_23971_: bool;
    var phi_23986_: bool;
    var phi_24001_: bool;
    var phi_3576_: type_32_;
    var phi_3639_: u32;
    var phi_3642_: type_32_;
    var phi_3744_: type_32_;
    var phi_3640_: u32;
    var phi_3643_: type_32_;
    var phi_34330_: bool;
    var phi_3868_: array<f32, 2>;
    var phi_3880_: u32;
    var phi_3944_: array<f32, 2>;
    var phi_24016_: bool;
    var phi_4059_: bool;
    var phi_24076_: bool;
    var phi_4024_: bool;
    var phi_4065_: f32;
    var phi_4066_: f32;
    var phi_4067_: f32;
    var phi_4068_: bool;
    var phi_4074_: f32;
    var phi_4076_: f32;
    var phi_4077_: bool;
    var phi_24136_: bool;
    var phi_4193_: bool;
    var phi_24196_: bool;
    var phi_4158_: bool;
    var phi_4199_: f32;
    var phi_4200_: f32;
    var phi_4201_: f32;
    var phi_4202_: bool;
    var phi_4208_: f32;
    var phi_4210_: f32;
    var phi_4211_: bool;
    var phi_24256_: bool;
    var phi_4343_: bool;
    var phi_24316_: bool;
    var phi_4308_: bool;
    var phi_4349_: f32;
    var phi_4350_: f32;
    var phi_4351_: f32;
    var phi_4352_: bool;
    var phi_4358_: f32;
    var phi_4360_: f32;
    var phi_4361_: bool;
    var phi_24376_: bool;
    var phi_4479_: bool;
    var phi_24436_: bool;
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
    var phi_34362_: bool;
    var phi_4619_: bool;
    var phi_4624_: bool;
    var phi_34455_: bool;
    var phi_4625_: array<f32, 2>;
    var phi_4626_: f32;
    var phi_4627_: f32;
    var phi_4628_: array<f32, 2>;
    var phi_4629_: bool;
    var phi_34454_: bool;
    var phi_3498_: array<f32, 2>;
    var phi_3500_: f32;
    var phi_3502_: f32;
    var phi_3504_: array<f32, 2>;
    var phi_3506_: bool;
    var phi_34453_: bool;
    var local_24_: bool;
    var phi_4992_: type_15_;
    var phi_4995_: f32;
    var phi_4997_: f32;
    var phi_4993_: type_15_;
    var phi_5020_: type_15_;
    var phi_24496_: bool;
    var phi_24556_: bool;
    var phi_24616_: bool;
    var phi_24676_: bool;
    var phi_5185_: bool;
    var phi_5239_: bool;
    var phi_4996_: f32;
    var phi_4998_: f32;
    var phi_34459_: bool;
    var local_25_: f32;
    var local_26_: f32;
    var phi_34506_: bool;
    var phi_5375_: type_15_;
    var phi_5378_: f32;
    var phi_5380_: f32;
    var phi_5376_: type_15_;
    var phi_5403_: type_15_;
    var phi_24736_: bool;
    var phi_24796_: bool;
    var phi_24856_: bool;
    var phi_24916_: bool;
    var phi_5569_: bool;
    var phi_5623_: bool;
    var phi_5379_: f32;
    var phi_5381_: f32;
    var phi_34502_: bool;
    var local_27_: f32;
    var local_28_: f32;
    var phi_34546_: bool;
    var phi_24976_: bool;
    var phi_24991_: bool;
    var phi_25006_: bool;
    var local_29_: f32;
    var local_30_: f32;
    var phi_25021_: bool;
    var local_31_: f32;
    var local_32_: f32;
    var local_33_: f32;
    var local_34_: f32;
    var local_35_: array<f32, 2>;
    var local_36_: array<f32, 2>;
    var local_37_: array<f32, 2>;
    var local_38_: array<f32, 2>;
    var phi_34545_: bool;
    var phi_34544_: bool;
    var phi_34543_: bool;
    var phi_34587_: bool;
    var phi_34586_: bool;
    var phi_34585_: bool;
    var phi_7861_: type_15_;
    var phi_7862_: type_15_;
    var phi_7885_: type_15_;
    var phi_7903_: f32;
    var phi_7914_: f32;
    var phi_7925_: f32;
    var phi_7936_: f32;
    var phi_7947_: f32;
    var phi_8054_: bool;
    var phi_8060_: f32;
    var phi_8061_: f32;
    var phi_8068_: f32;
    var phi_8093_: bool;
    var phi_8099_: f32;
    var phi_8100_: f32;
    var phi_8107_: f32;
    var phi_8132_: bool;
    var phi_8138_: f32;
    var phi_8139_: f32;
    var phi_8146_: f32;
    var phi_8171_: bool;
    var phi_8177_: f32;
    var phi_8178_: f32;
    var phi_8185_: f32;
    var phi_8210_: bool;
    var phi_8216_: f32;
    var phi_8217_: f32;
    var phi_8224_: f32;
    var phi_8319_: f32;
    var phi_8330_: f32;
    var phi_8400_: bool;
    var phi_8447_: bool;
    var phi_8464_: bool;
    var phi_8476_: f32;
    var phi_8488_: f32;
    var phi_8499_: bool;
    var phi_25036_: bool;
    var phi_25096_: bool;
    var phi_25156_: bool;
    var phi_25216_: bool;
    var phi_8989_: f32;
    var phi_9002_: f32;
    var phi_9016_: bool;
    var phi_9033_: bool;
    var phi_9041_: type_28_;
    var phi_9082_: u32;
    var phi_9085_: type_28_;
    var phi_9125_: type_28_;
    var phi_9083_: u32;
    var phi_9086_: type_28_;
    var phi_34559_: bool;
    var local_39_: type_28_;
    var local_40_: type_28_;
    var phi_9161_: u32;
    var phi_9182_: f32;
    var phi_9196_: bool;
    var phi_9227_: bool;
    var phi_9254_: bool;
    var phi_34712_: bool;
    var phi_34711_: bool;
    var phi_34710_: bool;
    var phi_16687_: f32;
    var phi_16698_: f32;
    var phi_16709_: f32;
    var phi_16720_: f32;
    var phi_16731_: f32;
    var phi_16838_: bool;
    var phi_16844_: f32;
    var phi_16845_: f32;
    var phi_16852_: f32;
    var phi_16877_: bool;
    var phi_16883_: f32;
    var phi_16884_: f32;
    var phi_16891_: f32;
    var phi_16916_: bool;
    var phi_16922_: f32;
    var phi_16923_: f32;
    var phi_16930_: f32;
    var phi_16955_: bool;
    var phi_16961_: f32;
    var phi_16962_: f32;
    var phi_16969_: f32;
    var phi_16994_: bool;
    var phi_17000_: f32;
    var phi_17001_: f32;
    var phi_17008_: f32;
    var phi_17118_: f32;
    var phi_17129_: f32;
    var phi_17199_: bool;
    var phi_17225_: u32;
    var phi_17235_: type_15_;
    var phi_17238_: f32;
    var phi_17236_: type_15_;
    var phi_17261_: type_15_;
    var phi_17306_: bool;
    var phi_25396_: bool;
    var phi_25456_: bool;
    var phi_25516_: bool;
    var phi_25576_: bool;
    var phi_25636_: bool;
    var phi_25696_: bool;
    var phi_25756_: bool;
    var phi_25816_: bool;
    var phi_18056_: bool;
    var phi_17239_: f32;
    var phi_34724_: bool;
    var local_41_: f32;
    var phi_18114_: bool;
    var phi_35680_: bool;
    var phi_35679_: bool;
    var phi_34906_: bool;
    var phi_9272_: type_15_;
    var phi_9273_: type_15_;
    var phi_9296_: type_15_;
    var phi_9314_: f32;
    var phi_9325_: f32;
    var phi_9336_: f32;
    var phi_9347_: f32;
    var phi_9358_: f32;
    var phi_9465_: bool;
    var phi_9471_: f32;
    var phi_9472_: f32;
    var phi_9479_: f32;
    var phi_9504_: bool;
    var phi_9510_: f32;
    var phi_9511_: f32;
    var phi_9518_: f32;
    var phi_9543_: bool;
    var phi_9549_: f32;
    var phi_9550_: f32;
    var phi_9557_: f32;
    var phi_9582_: bool;
    var phi_9588_: f32;
    var phi_9589_: f32;
    var phi_9596_: f32;
    var phi_9621_: bool;
    var phi_9627_: f32;
    var phi_9628_: f32;
    var phi_9635_: f32;
    var phi_9745_: f32;
    var phi_9756_: f32;
    var phi_9826_: bool;
    var phi_9853_: f32;
    var phi_9873_: f32;
    var phi_34878_: bool;
    var phi_9904_: type_15_;
    var phi_9907_: f32;
    var phi_9909_: bool;
    var phi_9905_: type_15_;
    var phi_9932_: type_15_;
    var phi_9955_: f32;
    var phi_9983_: bool;
    var phi_25996_: bool;
    var phi_26056_: bool;
    var phi_26116_: bool;
    var phi_26176_: bool;
    var phi_26236_: bool;
    var phi_26296_: bool;
    var phi_26356_: bool;
    var phi_26416_: bool;
    var phi_10733_: bool;
    var phi_10781_: bool;
    var phi_10797_: bool;
    var phi_10813_: bool;
    var phi_10830_: bool;
    var phi_10842_: f32;
    var phi_10854_: f32;
    var phi_26596_: bool;
    var phi_26656_: bool;
    var phi_26776_: bool;
    var phi_26836_: bool;
    var phi_26896_: bool;
    var phi_11350_: f32;
    var phi_11358_: f32;
    var phi_11487_: bool;
    var phi_11493_: f32;
    var phi_11494_: f32;
    var phi_11413_: f32;
    var phi_11415_: f32;
    var phi_11416_: bool;
    var phi_11436_: f32;
    var phi_11495_: f32;
    var phi_11498_: type_28_;
    var phi_26956_: bool;
    var phi_11615_: bool;
    var phi_27016_: bool;
    var phi_11580_: bool;
    var phi_11621_: f32;
    var phi_11622_: f32;
    var phi_11623_: f32;
    var phi_11624_: bool;
    var phi_11630_: f32;
    var phi_11632_: f32;
    var phi_11633_: bool;
    var phi_11638_: f32;
    var phi_11676_: u32;
    var phi_11677_: u32;
    var phi_34869_: bool;
    var phi_11736_: type_28_;
    var phi_11764_: u32;
    var phi_11767_: type_28_;
    var phi_11807_: type_28_;
    var phi_11765_: u32;
    var phi_11768_: type_28_;
    var phi_35160_: bool;
    var local_42_: type_28_;
    var local_43_: type_28_;
    var phi_11843_: u32;
    var phi_35175_: bool;
    var phi_11865_: type_34_;
    var phi_11868_: f32;
    var phi_11870_: f32;
    var phi_11872_: bool;
    var phi_11874_: bool;
    var phi_11901_: type_35_;
    var phi_11866_: type_34_;
    var phi_11915_: type_35_;
    var phi_12047_: bool;
    var phi_12053_: f32;
    var phi_12054_: f32;
    var phi_11973_: f32;
    var phi_11975_: f32;
    var phi_11976_: bool;
    var phi_11996_: f32;
    var phi_12055_: f32;
    var phi_12056_: f32;
    var phi_12077_: f32;
    var phi_12105_: bool;
    var phi_27076_: bool;
    var phi_27136_: bool;
    var phi_27196_: bool;
    var phi_27256_: bool;
    var phi_27316_: bool;
    var phi_27376_: bool;
    var phi_27436_: bool;
    var phi_27496_: bool;
    var phi_12855_: bool;
    var phi_28036_: bool;
    var phi_13693_: bool;
    var phi_28096_: bool;
    var phi_13658_: bool;
    var phi_13699_: f32;
    var phi_13700_: f32;
    var phi_13701_: f32;
    var phi_13702_: bool;
    var phi_13708_: f32;
    var phi_13710_: f32;
    var phi_13711_: bool;
    var phi_13716_: f32;
    var phi_13754_: u32;
    var phi_13755_: u32;
    var phi_35166_: bool;
    var phi_13811_: bool;
    var phi_13829_: bool;
    var phi_35316_: bool;
    var phi_13831_: f32;
    var phi_13832_: f32;
    var phi_13833_: bool;
    var phi_13834_: bool;
    var phi_35315_: bool;
    var phi_11869_: f32;
    var phi_11871_: f32;
    var phi_11873_: bool;
    var phi_11875_: bool;
    var phi_35314_: bool;
    var phi_35323_: bool;
    var phi_13835_: f32;
    var phi_13836_: bool;
    var phi_35322_: bool;
    var phi_13837_: f32;
    var phi_13838_: bool;
    var phi_35321_: bool;
    var phi_9908_: f32;
    var phi_9910_: bool;
    var phi_35320_: bool;
    var phi_13852_: f32;
    var local_44_: f32;
    var phi_13880_: bool;
    var phi_28156_: bool;
    var phi_28216_: bool;
    var phi_28276_: bool;
    var phi_28336_: bool;
    var phi_28396_: bool;
    var phi_28456_: bool;
    var phi_28516_: bool;
    var phi_28576_: bool;
    var phi_14630_: bool;
    var phi_14678_: bool;
    var phi_14694_: bool;
    var phi_14710_: bool;
    var phi_14727_: bool;
    var phi_14739_: f32;
    var phi_14751_: f32;
    var phi_28756_: bool;
    var phi_28816_: bool;
    var phi_28936_: bool;
    var phi_28996_: bool;
    var phi_29056_: bool;
    var phi_15247_: f32;
    var phi_15255_: f32;
    var phi_15384_: bool;
    var phi_15390_: f32;
    var phi_15391_: f32;
    var phi_15310_: f32;
    var phi_15312_: f32;
    var phi_15313_: bool;
    var phi_15333_: f32;
    var phi_15392_: f32;
    var phi_15395_: type_28_;
    var local_45_: f32;
    var local_46_: f32;
    var local_47_: f32;
    var phi_29116_: bool;
    var phi_15511_: bool;
    var phi_29176_: bool;
    var phi_15476_: bool;
    var phi_15517_: f32;
    var phi_15518_: f32;
    var phi_15519_: f32;
    var phi_15520_: bool;
    var phi_15526_: f32;
    var phi_15528_: f32;
    var phi_15529_: bool;
    var phi_15534_: f32;
    var phi_15572_: u32;
    var phi_15573_: u32;
    var phi_35324_: bool;
    var phi_16626_: bool;
    var phi_16657_: bool;
    var phi_15652_: u32;
    var phi_15662_: type_15_;
    var phi_15665_: f32;
    var phi_15663_: type_15_;
    var phi_15688_: type_15_;
    var phi_15733_: bool;
    var phi_29236_: bool;
    var phi_29296_: bool;
    var phi_29356_: bool;
    var phi_29416_: bool;
    var phi_29476_: bool;
    var phi_29536_: bool;
    var phi_29596_: bool;
    var phi_29656_: bool;
    var phi_16483_: bool;
    var phi_15666_: f32;
    var phi_35559_: bool;
    var local_48_: f32;
    var phi_16541_: bool;
    var phi_35671_: bool;
    var phi_35662_: bool;
    var phi_35661_: bool;
    var phi_35660_: bool;
    var phi_35659_: bool;
    var phi_35678_: bool;
    var phi_18135_: type_15_;
    var phi_18136_: type_15_;
    var phi_18159_: type_15_;
    var phi_35674_: bool;
    var phi_18331_: type_15_;
    var phi_18332_: type_15_;
    var phi_18355_: type_15_;
    var phi_18726_: f32;
    var phi_18737_: f32;
    var phi_18748_: f32;
    var phi_29836_: bool;
    var phi_29851_: bool;
    var phi_18855_: bool;
    var phi_18875_: bool;
    var phi_35683_: bool;
    var phi_35703_: bool;
    var phi_35702_: bool;
    var phi_18986_: type_15_;
    var phi_18987_: type_15_;
    var phi_19010_: type_15_;
    var phi_35698_: bool;
    var phi_18923_: type_15_;
    var phi_18924_: type_15_;
    var phi_18947_: type_15_;
    var phi_35704_: bool;
    var phi_36557_: bool;
    var phi_29866_: bool;
    var phi_29881_: bool;
    var phi_29896_: bool;
    var phi_29911_: bool;
    var phi_29926_: bool;
    var phi_29941_: bool;
    var phi_29956_: bool;
    var phi_29971_: bool;
    var phi_29986_: bool;
    var phi_30001_: bool;
    var phi_30016_: bool;
    var phi_30031_: bool;
    var phi_30046_: bool;
    var phi_30061_: bool;
    var phi_30076_: bool;
    var phi_30091_: bool;
    var phi_30106_: bool;
    var phi_30121_: bool;
    var phi_30136_: bool;
    var phi_30151_: bool;
    var phi_30166_: bool;
    var phi_30181_: bool;
    var phi_30196_: bool;
    var phi_30211_: bool;
    var phi_30226_: bool;
    var phi_30241_: bool;
    var phi_30256_: bool;
    var phi_30271_: bool;
    var phi_30286_: bool;
    var phi_30301_: bool;
    var phi_30316_: bool;
    var phi_30331_: bool;
    var phi_30346_: bool;
    var phi_30361_: bool;
    var phi_30376_: bool;
    var phi_30391_: bool;
    var phi_30406_: bool;
    var phi_30421_: bool;
    var phi_30436_: bool;
    var phi_30451_: bool;
    var phi_30466_: bool;
    var phi_30481_: bool;
    var phi_30496_: bool;
    var phi_30511_: bool;
    var phi_30526_: bool;
    var phi_30541_: bool;
    var phi_30556_: bool;
    var phi_30571_: bool;
    var phi_30586_: bool;
    var phi_30601_: bool;
    var phi_30616_: bool;
    var phi_30631_: bool;
    var phi_30646_: bool;
    var phi_30661_: bool;
    var phi_30676_: bool;
    var phi_30691_: bool;
    var phi_30706_: bool;
    var phi_30721_: bool;
    var phi_30736_: bool;
    var phi_30751_: bool;
    var phi_30766_: bool;
    var phi_30781_: bool;
    var phi_30796_: bool;
    var phi_30811_: bool;
    var phi_19515_: bool;
    var phi_19517_: bool;
    var phi_19518_: bool;
    var phi_19519_: bool;
    var phi_19520_: bool;
    var phi_30826_: bool;
    var phi_30841_: bool;
    var phi_30856_: bool;
    var phi_30871_: bool;
    var phi_19898_: type_38_;
    var phi_30886_: bool;
    var phi_30901_: bool;
    var phi_30916_: bool;
    var phi_30931_: bool;
    var phi_30946_: bool;
    var phi_30961_: bool;
    var phi_30976_: bool;
    var phi_30991_: bool;
    var phi_31006_: bool;
    var phi_31021_: bool;
    var phi_31036_: bool;
    var phi_31051_: bool;
    var phi_31066_: bool;
    var phi_31081_: bool;
    var phi_31096_: bool;
    var phi_31111_: bool;
    var phi_31126_: bool;
    var phi_31141_: bool;
    var phi_31156_: bool;
    var phi_31171_: bool;
    var phi_31186_: bool;
    var phi_31201_: bool;
    var phi_31216_: bool;
    var phi_31231_: bool;
    var phi_31246_: bool;
    var phi_31261_: bool;
    var phi_31276_: bool;
    var phi_31291_: bool;
    var phi_31306_: bool;
    var phi_31321_: bool;
    var phi_31336_: bool;
    var phi_31351_: bool;
    var phi_31366_: bool;
    var phi_31381_: bool;
    var phi_31396_: bool;
    var phi_31411_: bool;
    var phi_31426_: bool;
    var phi_31441_: bool;
    var phi_31456_: bool;
    var phi_31471_: bool;
    var phi_31486_: bool;
    var phi_31501_: bool;
    var phi_31516_: bool;
    var phi_31531_: bool;
    var phi_31546_: bool;
    var phi_31561_: bool;
    var phi_31576_: bool;
    var phi_31591_: bool;
    var phi_31606_: bool;
    var phi_31621_: bool;
    var phi_31636_: bool;
    var phi_31651_: bool;
    var phi_31666_: bool;
    var phi_31681_: bool;
    var phi_31696_: bool;
    var phi_31711_: bool;
    var phi_31726_: bool;
    var phi_31741_: bool;
    var phi_31756_: bool;
    var phi_31771_: bool;
    var phi_31786_: bool;
    var phi_31801_: bool;
    var phi_31816_: bool;
    var phi_31831_: bool;
    var phi_20344_: bool;
    var phi_20346_: bool;
    var phi_20347_: bool;
    var phi_20348_: bool;
    var phi_20349_: bool;
    var phi_31846_: bool;
    var phi_31861_: bool;
    var phi_31876_: bool;
    var phi_31891_: bool;
    var phi_20727_: type_38_;
    var phi_20744_: bool;
    var phi_20746_: bool;
    var phi_20747_: bool;
    var phi_20748_: bool;
    var phi_20749_: bool;
    var phi_31906_: bool;
    var phi_31921_: bool;
    var phi_31936_: bool;
    var phi_31951_: bool;
    var phi_31966_: bool;
    var phi_31981_: bool;
    var phi_31996_: bool;
    var phi_32011_: bool;
    var phi_32026_: bool;
    var phi_32041_: bool;
    var phi_32056_: bool;
    var phi_32071_: bool;
    var phi_32086_: bool;
    var phi_32101_: bool;
    var phi_32116_: bool;
    var phi_32131_: bool;
    var phi_32146_: bool;
    var phi_32161_: bool;
    var phi_32176_: bool;
    var phi_32191_: bool;
    var phi_32206_: bool;
    var phi_32221_: bool;
    var phi_32236_: bool;
    var phi_32251_: bool;
    var phi_32266_: bool;
    var phi_32281_: bool;
    var phi_32296_: bool;
    var phi_32311_: bool;
    var phi_32326_: bool;
    var phi_32341_: bool;
    var phi_32356_: bool;
    var phi_32371_: bool;
    var phi_32386_: bool;
    var phi_32401_: bool;
    var phi_32416_: bool;
    var phi_32431_: bool;
    var phi_32446_: bool;
    var phi_32461_: bool;
    var phi_32476_: bool;
    var phi_32491_: bool;
    var phi_32506_: bool;
    var phi_32521_: bool;
    var phi_32536_: bool;
    var phi_32551_: bool;
    var phi_32806_: bool;
    var phi_32821_: bool;
    var phi_32836_: bool;
    var phi_32851_: bool;
    var phi_21359_: bool;
    var phi_21361_: bool;
    var phi_21362_: bool;
    var phi_21363_: bool;
    var phi_21364_: bool;
    var phi_32866_: bool;
    var phi_32881_: bool;
    var phi_32896_: bool;
    var phi_32911_: bool;
    var phi_21796_: type_31_;
    var phi_32926_: bool;
    var phi_32941_: bool;
    var phi_32956_: bool;
    var phi_32971_: bool;
    var phi_32986_: bool;
    var phi_33001_: bool;
    var phi_33016_: bool;
    var phi_33031_: bool;
    var phi_33046_: bool;
    var phi_33061_: bool;
    var phi_33076_: bool;
    var phi_33091_: bool;
    var phi_33106_: bool;
    var phi_33121_: bool;
    var phi_33136_: bool;
    var phi_33151_: bool;
    var phi_33166_: bool;
    var phi_33181_: bool;
    var phi_33196_: bool;
    var phi_33211_: bool;
    var phi_33226_: bool;
    var phi_33241_: bool;
    var phi_33256_: bool;
    var phi_33271_: bool;
    var phi_33286_: bool;
    var phi_33301_: bool;
    var phi_33316_: bool;
    var phi_33331_: bool;
    var phi_33346_: bool;
    var phi_33361_: bool;
    var phi_33376_: bool;
    var phi_33391_: bool;
    var phi_33406_: bool;
    var phi_33421_: bool;
    var phi_33436_: bool;
    var phi_33451_: bool;
    var phi_33466_: bool;
    var phi_33481_: bool;
    var phi_33496_: bool;
    var phi_33511_: bool;
    var phi_33526_: bool;
    var phi_33541_: bool;
    var phi_33556_: bool;
    var phi_33571_: bool;
    var phi_33826_: bool;
    var phi_33841_: bool;
    var phi_33856_: bool;
    var phi_33871_: bool;
    var phi_22400_: bool;
    var phi_22402_: bool;
    var phi_22403_: bool;
    var phi_22404_: bool;
    var phi_22405_: bool;
    var phi_33886_: bool;
    var phi_33901_: bool;
    var phi_33916_: bool;
    var phi_33931_: bool;
    var phi_22837_: type_31_;
    var phi_22891_: type_30_;
    var phi_22892_: type_30_;
    var phi_23048_: bool;
    var phi_23064_: bool;
    var phi_23080_: bool;
    var phi_23096_: bool;
    var phi_23134_: bool;
    var phi_23150_: bool;
    var phi_23166_: bool;
    var phi_23182_: bool;
    var phi_23214_: bool;
    var phi_23230_: bool;
    var phi_23246_: bool;
    var phi_23262_: bool;
    var phi_23294_: bool;
    var phi_23310_: bool;
    var phi_23326_: bool;
    var phi_23342_: bool;
    var phi_23374_: bool;
    var phi_23390_: bool;
    var phi_23406_: bool;
    var phi_23422_: bool;
    var phi_23454_: bool;
    var phi_23470_: bool;
    var phi_23486_: bool;
    var phi_23502_: bool;
    var phi_23526_: type_15_;
    var phi_23527_: type_15_;
    var phi_23550_: type_15_;
    var phi_23575_: bool;
    var phi_23592_: f32;
    var phi_23617_: bool;
    var phi_23634_: f32;
    var phi_35708_: bool;
    var phi_36581_: bool;
    var phi_36580_: bool;
    var phi_36579_: bool;
    var phi_36578_: bool;
    var phi_36577_: bool;
    var phi_36576_: bool;
    var phi_36583_: bool;
    var phi_36582_: bool;
    var local_49_: u32;
    var local_50_: u32;
    var local_51_: f32;
    var local_52_: bool;

    switch bitcast<i32>(0u) {
        default: {
            let _e158_ = wg_id_1_;
            let _e159_ = num_wg_1_;
            let _e160_ = lid_1_;
            let _e162_ = arrayLength((&core_inputs.member));
            let _e164_ = arrayLength((&quad_params.member));
            let _e166_ = arrayLength((&tile_checkpoints.member));
            let _e168_ = arrayLength((&feature_outputs.member));
            let _e169_ = (_e160_ & 255u);
            let _e170_ = (_e169_ & 127u);
            let _e173_ = ((_e169_ >> bitcast<u32>(7i)) != 0u);
            let _e178_ = select(1u, _e159_.y, (1u < _e159_.y));
            let _e182_ = uniforms.member.member_2_;
            let _e184_ = select(1u, _e182_, (1u < _e182_));
            let _e187_ = uniforms.member.member_7_;
            let _e188_ = (_e187_ == 4294967295u);
            if _e188_ {
                phi_303_ = select(16u, _e158_.x, (_e158_.x < _e184_));
            } else {
                let _e189_ = (_e158_.x < 4u);
                if _e189_ {
                    local_8_ = array<array<u32, 4>, 4>(array<u32, 4>(0u, 4u, 6u, 8u), array<u32, 4>(1u, 5u, 9u, 16u), array<u32, 4>(2u, 7u, 16u, 16u), array<u32, 4>(3u, 16u, 16u, 16u));
                    let _e190_ = (_e187_ & 3u);
                    if (_e190_ < 4u) {
                    } else {
                        break;
                    }
                    if _e189_ {
                    } else {
                        break;
                    }
                    let _e194_ = local_8_[_e190_][_e158_.x];
                    phi_302_ = select(16u, _e194_, (_e194_ < _e184_));
                } else {
                    phi_302_ = 16u;
                }
                let _e198_ = phi_302_;
                phi_303_ = _e198_;
            }
            let _e202_ = phi_303_;
            if (_e202_ >= _e184_) {
            } else {
                let _e205_ = ((_e158_.z * _e184_) + _e202_);
                if _e188_ {
                    phi_320_ = false;
                } else {
                    phi_320_ = ((_e187_ & 1073741824u) != 0u);
                }
                let _e209_ = phi_320_;
                if _e188_ {
                    phi_328_ = false;
                } else {
                    phi_328_ = ((_e187_ & 3u) == 0u);
                }
                let _e213_ = phi_328_;
                let _e216_ = uniforms.member.member;
                if (_e216_ > 0u) {
                    phi_338_ = (((_e216_ - 1u) >> bitcast<u32>(8i)) + 1u);
                } else {
                    phi_338_ = 0u;
                }
                let _e223_ = phi_338_;
                let _e226_ = uniforms.member.member_1_;
                let _e228_ = select(1u, _e226_, (1u < _e226_));
                let _e230_ = select(_e226_, _e223_, (_e223_ < _e226_));
                let _e233_ = uniforms.member.member_4_;
                let _e235_ = select(_e233_, _e230_, (_e230_ < _e233_));
                let _e236_ = (_e178_ <= 1u);
                if _e236_ {
                    phi_436_ = _e235_;
                    phi_437_ = _e230_;
                } else {
                    if (_e230_ > _e235_) {
                        phi_369_ = (_e230_ - _e235_);
                    } else {
                        phi_369_ = 0u;
                    }
                    let _e240_ = phi_369_;
                    if (_e240_ == 0u) {
                        phi_422_ = 0u;
                    } else {
                        if _e236_ {
                            phi_421_ = _e240_;
                        } else {
                            let _e242_ = (_e240_ - 1u);
                            phi_380_ = 1u;
                            phi_383_ = _e178_;
                            loop {
                                let _e244_ = phi_380_;
                                let _e246_ = phi_383_;
                                local_49_ = _e244_;
                                local_50_ = _e246_;
                                let _e249_ = (_e246_ <= (_e242_ >> bitcast<u32>(1i)));
                                if _e249_ {
                                    phi_381_ = (_e244_ << bitcast<u32>(1i));
                                    phi_384_ = (_e246_ << bitcast<u32>(1i));
                                } else {
                                    phi_381_ = u32();
                                    phi_384_ = u32();
                                }
                                let _e255_ = phi_381_;
                                let _e257_ = phi_384_;
                                phi_380_ = _e255_;
                                phi_383_ = _e257_;
                                if !(_e249_) {
                                    break;
                                }
                            }
                            phi_395_ = 0u;
                            let _e9756_ = local_49_;
                            phi_398_ = _e9756_;
                            phi_400_ = _e242_;
                            let _e9760_ = local_50_;
                            phi_402_ = _e9760_;
                            loop {
                                let _e260_ = phi_395_;
                                let _e262_ = phi_398_;
                                let _e264_ = phi_400_;
                                let _e266_ = phi_402_;
                                local_21_ = _e260_;
                                let _e267_ = (_e262_ > 0u);
                                if _e267_ {
                                    if (_e264_ >= _e266_) {
                                        phi_416_ = (_e260_ | _e262_);
                                        phi_417_ = (_e264_ - _e266_);
                                    } else {
                                        phi_416_ = _e260_;
                                        phi_417_ = _e264_;
                                    }
                                    let _e272_ = phi_416_;
                                    let _e274_ = phi_417_;
                                    phi_396_ = _e272_;
                                    phi_399_ = (_e262_ >> bitcast<u32>(1i));
                                    phi_401_ = _e274_;
                                    phi_403_ = (_e266_ >> bitcast<u32>(1i));
                                } else {
                                    phi_396_ = u32();
                                    phi_399_ = u32();
                                    phi_401_ = u32();
                                    phi_403_ = u32();
                                }
                                let _e280_ = phi_396_;
                                let _e282_ = phi_399_;
                                let _e284_ = phi_401_;
                                let _e286_ = phi_403_;
                                phi_395_ = _e280_;
                                phi_398_ = _e282_;
                                phi_400_ = _e284_;
                                phi_402_ = _e286_;
                                if !(_e267_) {
                                    break;
                                }
                            }
                            let _e289_ = local_21_;
                            phi_421_ = (_e289_ + 1u);
                        }
                        let _e292_ = phi_421_;
                        phi_422_ = _e292_;
                    }
                    let _e294_ = phi_422_;
                    let _e296_ = (_e235_ + (_e158_.y * _e294_));
                    let _e298_ = select(_e296_, _e230_, (_e230_ < _e296_));
                    let _e299_ = (_e298_ + _e294_);
                    phi_436_ = _e298_;
                    phi_437_ = select(_e299_, _e230_, (_e230_ < _e299_));
                }
                let _e303_ = phi_436_;
                let _e305_ = phi_437_;
                let _e306_ = (_e205_ * 25u);
                let _e307_ = (_e306_ < _e164_);
                if _e307_ {
                    if _e307_ {
                    } else {
                        break;
                    }
                    let _e310_ = quad_params.member[_e306_];
                    phi_451_ = _e310_;
                } else {
                    phi_451_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e312_ = phi_451_;
                let _e313_ = (_e306_ + 1u);
                let _e314_ = (_e313_ < _e164_);
                if _e314_ {
                    if _e314_ {
                    } else {
                        break;
                    }
                    let _e317_ = quad_params.member[_e313_];
                    phi_464_ = _e317_;
                } else {
                    phi_464_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e319_ = phi_464_;
                let _e320_ = (_e306_ + 2u);
                let _e321_ = (_e320_ < _e164_);
                if _e321_ {
                    if _e321_ {
                    } else {
                        break;
                    }
                    let _e324_ = quad_params.member[_e320_];
                    phi_477_ = _e324_;
                } else {
                    phi_477_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e326_ = phi_477_;
                let _e327_ = (_e306_ + 3u);
                let _e328_ = (_e327_ < _e164_);
                if _e328_ {
                    if _e328_ {
                    } else {
                        break;
                    }
                    let _e331_ = quad_params.member[_e327_];
                    phi_490_ = _e331_;
                } else {
                    phi_490_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e333_ = phi_490_;
                let _e334_ = (_e306_ + 4u);
                let _e335_ = (_e334_ < _e164_);
                if _e335_ {
                    if _e335_ {
                    } else {
                        break;
                    }
                    let _e338_ = quad_params.member[_e334_];
                    phi_503_ = _e338_;
                } else {
                    phi_503_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e340_ = phi_503_;
                let _e341_ = (_e306_ + 5u);
                let _e342_ = (_e341_ < _e164_);
                if _e342_ {
                    if _e342_ {
                    } else {
                        break;
                    }
                    let _e345_ = quad_params.member[_e341_];
                    phi_516_ = _e345_;
                } else {
                    phi_516_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e347_ = phi_516_;
                let _e348_ = (_e306_ + 6u);
                let _e349_ = (_e348_ < _e164_);
                if _e349_ {
                    if _e349_ {
                    } else {
                        break;
                    }
                    let _e352_ = quad_params.member[_e348_];
                    phi_529_ = _e352_;
                } else {
                    phi_529_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e354_ = phi_529_;
                let _e355_ = (_e306_ + 7u);
                let _e356_ = (_e355_ < _e164_);
                if _e356_ {
                    if _e356_ {
                    } else {
                        break;
                    }
                    let _e359_ = quad_params.member[_e355_];
                    phi_542_ = _e359_;
                } else {
                    phi_542_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e361_ = phi_542_;
                let _e362_ = (_e306_ + 8u);
                let _e363_ = (_e362_ < _e164_);
                if _e363_ {
                    if _e363_ {
                    } else {
                        break;
                    }
                    let _e366_ = quad_params.member[_e362_];
                    phi_555_ = _e366_;
                } else {
                    phi_555_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e368_ = phi_555_;
                let _e369_ = (_e306_ + 9u);
                let _e370_ = (_e369_ < _e164_);
                if _e370_ {
                    if _e370_ {
                    } else {
                        break;
                    }
                    let _e373_ = quad_params.member[_e369_];
                    phi_568_ = _e373_;
                } else {
                    phi_568_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e375_ = phi_568_;
                let _e376_ = (_e306_ + 10u);
                let _e377_ = (_e376_ < _e164_);
                if _e377_ {
                    if _e377_ {
                    } else {
                        break;
                    }
                    let _e380_ = quad_params.member[_e376_];
                    phi_581_ = _e380_;
                } else {
                    phi_581_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e382_ = phi_581_;
                let _e383_ = (_e306_ + 11u);
                let _e384_ = (_e383_ < _e164_);
                if _e384_ {
                    if _e384_ {
                    } else {
                        break;
                    }
                    let _e387_ = quad_params.member[_e383_];
                    phi_594_ = _e387_;
                } else {
                    phi_594_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e389_ = phi_594_;
                let _e390_ = (_e306_ + 12u);
                let _e391_ = (_e390_ < _e164_);
                if _e391_ {
                    if _e391_ {
                    } else {
                        break;
                    }
                    let _e394_ = quad_params.member[_e390_];
                    phi_607_ = _e394_;
                } else {
                    phi_607_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e396_ = phi_607_;
                let _e397_ = (_e306_ + 13u);
                let _e398_ = (_e397_ < _e164_);
                if _e398_ {
                    if _e398_ {
                    } else {
                        break;
                    }
                    let _e401_ = quad_params.member[_e397_];
                    phi_620_ = _e401_;
                } else {
                    phi_620_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e403_ = phi_620_;
                let _e404_ = (_e306_ + 14u);
                let _e405_ = (_e404_ < _e164_);
                if _e405_ {
                    if _e405_ {
                    } else {
                        break;
                    }
                    let _e408_ = quad_params.member[_e404_];
                    phi_633_ = _e408_;
                } else {
                    phi_633_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e410_ = phi_633_;
                let _e411_ = (_e306_ + 15u);
                let _e412_ = (_e411_ < _e164_);
                if _e412_ {
                    if _e412_ {
                    } else {
                        break;
                    }
                    let _e415_ = quad_params.member[_e411_];
                    phi_646_ = _e415_;
                } else {
                    phi_646_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e417_ = phi_646_;
                let _e418_ = (_e306_ + 16u);
                let _e419_ = (_e418_ < _e164_);
                if _e419_ {
                    if _e419_ {
                    } else {
                        break;
                    }
                    let _e422_ = quad_params.member[_e418_];
                    phi_659_ = _e422_;
                } else {
                    phi_659_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e424_ = phi_659_;
                let _e425_ = (_e306_ + 17u);
                let _e426_ = (_e425_ < _e164_);
                if _e426_ {
                    if _e426_ {
                    } else {
                        break;
                    }
                    let _e429_ = quad_params.member[_e425_];
                    phi_672_ = _e429_;
                } else {
                    phi_672_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e431_ = phi_672_;
                let _e432_ = (_e306_ + 18u);
                let _e433_ = (_e432_ < _e164_);
                if _e433_ {
                    if _e433_ {
                    } else {
                        break;
                    }
                    let _e436_ = quad_params.member[_e432_];
                    phi_685_ = _e436_;
                } else {
                    phi_685_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e438_ = phi_685_;
                let _e439_ = (_e306_ + 19u);
                let _e440_ = (_e439_ < _e164_);
                if _e440_ {
                    if _e440_ {
                    } else {
                        break;
                    }
                    let _e443_ = quad_params.member[_e439_];
                    phi_698_ = _e443_;
                } else {
                    phi_698_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e445_ = phi_698_;
                let _e446_ = (_e306_ + 20u);
                let _e447_ = (_e446_ < _e164_);
                if _e447_ {
                    if _e447_ {
                    } else {
                        break;
                    }
                    let _e450_ = quad_params.member[_e446_];
                    phi_711_ = _e450_;
                } else {
                    phi_711_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e452_ = phi_711_;
                let _e453_ = (_e306_ + 21u);
                let _e454_ = (_e453_ < _e164_);
                if _e454_ {
                    if _e454_ {
                    } else {
                        break;
                    }
                    let _e457_ = quad_params.member[_e453_];
                    phi_724_ = _e457_;
                } else {
                    phi_724_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e459_ = phi_724_;
                let _e460_ = (_e306_ + 22u);
                let _e461_ = (_e460_ < _e164_);
                if _e461_ {
                    if _e461_ {
                    } else {
                        break;
                    }
                    let _e464_ = quad_params.member[_e460_];
                    phi_737_ = _e464_;
                } else {
                    phi_737_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e466_ = phi_737_;
                let _e468_ = ((_e306_ + 23u) < _e164_);
                if _e468_ {
                    if _e468_ {
                    } else {
                        break;
                    }
                }
                let _e469_ = (_e306_ + 24u);
                let _e470_ = (_e469_ < _e164_);
                if _e470_ {
                    if _e470_ {
                    } else {
                        break;
                    }
                    let _e473_ = quad_params.member[_e469_];
                    phi_763_ = _e473_;
                } else {
                    phi_763_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e475_ = phi_763_;
                let _e477_ = bitcast<u32>(_e459_.x);
                let _e478_ = (_e477_ & 2139095040u);
                if (_e478_ != 2139095040u) {
                    if (_e459_.x <= 0f) {
                        phi_797_ = 0f;
                    } else {
                        if (_e459_.x >= 1f) {
                            phi_796_ = 1f;
                        } else {
                            if (_e478_ == 2139095040u) {
                                phi_790_ = true;
                            } else {
                                phi_790_ = (_e478_ == 0u);
                            }
                            let _e487_ = phi_790_;
                            phi_796_ = bitcast<f32>(select(_e477_, 0u, _e487_));
                        }
                        let _e491_ = phi_796_;
                        phi_797_ = _e491_;
                    }
                    let _e493_ = phi_797_;
                    phi_804_ = _e493_;
                } else {
                    phi_804_ = select(0f, 1f, (_e477_ == 2139095040u));
                }
                let _e495_ = phi_804_;
                let _e497_ = bitcast<u32>(_e459_.y);
                let _e498_ = (_e497_ & 2139095040u);
                if (_e498_ != 2139095040u) {
                    if (_e459_.y <= 0f) {
                        phi_837_ = 0f;
                    } else {
                        if (_e459_.y >= 1f) {
                            phi_836_ = 1f;
                        } else {
                            if (_e498_ == 2139095040u) {
                                phi_830_ = true;
                            } else {
                                phi_830_ = (_e498_ == 0u);
                            }
                            let _e507_ = phi_830_;
                            phi_836_ = bitcast<f32>(select(_e497_, 0u, _e507_));
                        }
                        let _e511_ = phi_836_;
                        phi_837_ = _e511_;
                    }
                    let _e513_ = phi_837_;
                    phi_844_ = _e513_;
                } else {
                    phi_844_ = select(0f, 1f, (_e497_ == 2139095040u));
                }
                let _e515_ = phi_844_;
                let _e517_ = bitcast<u32>(_e459_.z);
                let _e518_ = (_e517_ & 2139095040u);
                if (_e518_ != 2139095040u) {
                    if (_e459_.z <= 0f) {
                        phi_877_ = 0f;
                    } else {
                        if (_e459_.z >= 1f) {
                            phi_876_ = 1f;
                        } else {
                            if (_e518_ == 2139095040u) {
                                phi_870_ = true;
                            } else {
                                phi_870_ = (_e518_ == 0u);
                            }
                            let _e527_ = phi_870_;
                            phi_876_ = bitcast<f32>(select(_e517_, 0u, _e527_));
                        }
                        let _e531_ = phi_876_;
                        phi_877_ = _e531_;
                    }
                    let _e533_ = phi_877_;
                    phi_884_ = _e533_;
                } else {
                    phi_884_ = select(0f, 1f, (_e517_ == 2139095040u));
                }
                let _e535_ = phi_884_;
                let _e537_ = bitcast<u32>(_e459_.w);
                let _e538_ = (_e537_ & 2139095040u);
                if (_e538_ != 2139095040u) {
                    if (_e459_.w <= 0f) {
                        phi_917_ = 0f;
                    } else {
                        if (_e459_.w >= 1f) {
                            phi_916_ = 1f;
                        } else {
                            if (_e538_ == 2139095040u) {
                                phi_910_ = true;
                            } else {
                                phi_910_ = (_e538_ == 0u);
                            }
                            let _e547_ = phi_910_;
                            phi_916_ = bitcast<f32>(select(_e537_, 0u, _e547_));
                        }
                        let _e551_ = phi_916_;
                        phi_917_ = _e551_;
                    }
                    let _e553_ = phi_917_;
                    phi_924_ = _e553_;
                } else {
                    phi_924_ = select(0f, 1f, (_e537_ == 2139095040u));
                }
                let _e555_ = phi_924_;
                let _e557_ = bitcast<u32>(_e417_.x);
                let _e558_ = (_e557_ & 2139095040u);
                if (_e558_ != 2139095040u) {
                    if (_e417_.x <= 0f) {
                        phi_957_ = 0f;
                    } else {
                        if (_e417_.x >= 1f) {
                            phi_956_ = 1f;
                        } else {
                            if (_e558_ == 2139095040u) {
                                phi_950_ = true;
                            } else {
                                phi_950_ = (_e558_ == 0u);
                            }
                            let _e567_ = phi_950_;
                            phi_956_ = bitcast<f32>(select(_e557_, 0u, _e567_));
                        }
                        let _e571_ = phi_956_;
                        phi_957_ = _e571_;
                    }
                    let _e573_ = phi_957_;
                    phi_964_ = _e573_;
                } else {
                    phi_964_ = select(0f, 1f, (_e557_ == 2139095040u));
                }
                let _e575_ = phi_964_;
                let _e577_ = bitcast<u32>(_e417_.y);
                let _e578_ = (_e577_ & 2139095040u);
                if (_e578_ != 2139095040u) {
                    if (_e417_.y <= 0f) {
                        phi_997_ = 0f;
                    } else {
                        if (_e417_.y >= 1f) {
                            phi_996_ = 1f;
                        } else {
                            if (_e578_ == 2139095040u) {
                                phi_990_ = true;
                            } else {
                                phi_990_ = (_e578_ == 0u);
                            }
                            let _e587_ = phi_990_;
                            phi_996_ = bitcast<f32>(select(_e577_, 0u, _e587_));
                        }
                        let _e591_ = phi_996_;
                        phi_997_ = _e591_;
                    }
                    let _e593_ = phi_997_;
                    phi_1004_ = _e593_;
                } else {
                    phi_1004_ = select(0f, 1f, (_e577_ == 2139095040u));
                }
                let _e595_ = phi_1004_;
                let _e597_ = bitcast<u32>(_e417_.z);
                let _e598_ = (_e597_ & 2139095040u);
                if (_e598_ != 2139095040u) {
                    if (_e417_.z <= 0f) {
                        phi_1037_ = 0f;
                    } else {
                        if (_e417_.z >= 1f) {
                            phi_1036_ = 1f;
                        } else {
                            if (_e598_ == 2139095040u) {
                                phi_1030_ = true;
                            } else {
                                phi_1030_ = (_e598_ == 0u);
                            }
                            let _e607_ = phi_1030_;
                            phi_1036_ = bitcast<f32>(select(_e597_, 0u, _e607_));
                        }
                        let _e611_ = phi_1036_;
                        phi_1037_ = _e611_;
                    }
                    let _e613_ = phi_1037_;
                    phi_1044_ = _e613_;
                } else {
                    phi_1044_ = select(0f, 1f, (_e597_ == 2139095040u));
                }
                let _e615_ = phi_1044_;
                let _e617_ = bitcast<u32>(_e417_.w);
                let _e618_ = (_e617_ & 2139095040u);
                if (_e618_ != 2139095040u) {
                    if (_e417_.w <= 0f) {
                        phi_1077_ = 0f;
                    } else {
                        if (_e417_.w >= 1f) {
                            phi_1076_ = 1f;
                        } else {
                            if (_e618_ == 2139095040u) {
                                phi_1070_ = true;
                            } else {
                                phi_1070_ = (_e618_ == 0u);
                            }
                            let _e627_ = phi_1070_;
                            phi_1076_ = bitcast<f32>(select(_e617_, 0u, _e627_));
                        }
                        let _e631_ = phi_1076_;
                        phi_1077_ = _e631_;
                    }
                    let _e633_ = phi_1077_;
                    phi_1084_ = _e633_;
                } else {
                    phi_1084_ = select(0f, 1f, (_e617_ == 2139095040u));
                }
                let _e635_ = phi_1084_;
                if _e236_ {
                    let _e640_ = (_e235_ > 0u);
                    if _e640_ {
                        let _e641_ = (_e235_ < _e228_);
                        let _e642_ = (_e169_ == 0u);
                        if _e642_ {
                            if _e641_ {
                                let _e645_ = (((_e205_ * _e228_) + _e235_) * 8u);
                                let _e646_ = (_e645_ < _e166_);
                                if _e646_ {
                                    if _e646_ {
                                    } else {
                                        break;
                                    }
                                    let _e649_ = tile_checkpoints.member[_e645_];
                                    phi_1120_ = _e649_;
                                } else {
                                    phi_1120_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e651_ = phi_1120_;
                                let _e653_ = bitcast<u32>(_e651_.x);
                                let _e654_ = (_e653_ & 2139095040u);
                                if (_e654_ == 2139095040u) {
                                    phi_1132_ = true;
                                } else {
                                    phi_1132_ = (_e654_ == 0u);
                                }
                                let _e658_ = phi_1132_;
                                let _e662_ = bitcast<u32>(_e651_.y);
                                let _e663_ = (_e662_ & 2139095040u);
                                if (_e663_ == 2139095040u) {
                                    phi_1149_ = true;
                                } else {
                                    phi_1149_ = (_e663_ == 0u);
                                }
                                let _e667_ = phi_1149_;
                                let _e671_ = bitcast<u32>(_e651_.z);
                                let _e672_ = (_e671_ & 2139095040u);
                                if (_e672_ == 2139095040u) {
                                    phi_1166_ = true;
                                } else {
                                    phi_1166_ = (_e672_ == 0u);
                                }
                                let _e676_ = phi_1166_;
                                let _e680_ = bitcast<u32>(_e651_.w);
                                let _e681_ = (_e680_ & 2139095040u);
                                if (_e681_ == 2139095040u) {
                                    phi_1183_ = true;
                                } else {
                                    phi_1183_ = (_e681_ == 0u);
                                }
                                let _e685_ = phi_1183_;
                                local = array<f32, 4>(bitcast<f32>(select(_e653_, 0u, _e658_)), bitcast<f32>(select(_e662_, 0u, _e667_)), bitcast<f32>(select(_e671_, 0u, _e676_)), bitcast<f32>(select(_e680_, 0u, _e685_)));
                                let _e689_ = (_e645_ + 1u);
                                let _e690_ = (_e689_ < _e166_);
                                if _e690_ {
                                    if _e690_ {
                                    } else {
                                        break;
                                    }
                                    let _e693_ = tile_checkpoints.member[_e689_];
                                    phi_1207_ = _e693_;
                                } else {
                                    phi_1207_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e695_ = phi_1207_;
                                let _e697_ = bitcast<u32>(_e695_.x);
                                let _e698_ = (_e697_ & 2139095040u);
                                if (_e698_ == 2139095040u) {
                                    phi_1219_ = true;
                                } else {
                                    phi_1219_ = (_e698_ == 0u);
                                }
                                let _e702_ = phi_1219_;
                                let _e706_ = bitcast<u32>(_e695_.y);
                                let _e707_ = (_e706_ & 2139095040u);
                                if (_e707_ == 2139095040u) {
                                    phi_1236_ = true;
                                } else {
                                    phi_1236_ = (_e707_ == 0u);
                                }
                                let _e711_ = phi_1236_;
                                let _e715_ = bitcast<u32>(_e695_.z);
                                let _e716_ = (_e715_ & 2139095040u);
                                if (_e716_ == 2139095040u) {
                                    phi_1253_ = true;
                                } else {
                                    phi_1253_ = (_e716_ == 0u);
                                }
                                let _e720_ = phi_1253_;
                                let _e724_ = bitcast<u32>(_e695_.w);
                                let _e725_ = (_e724_ & 2139095040u);
                                if (_e725_ == 2139095040u) {
                                    phi_1270_ = true;
                                } else {
                                    phi_1270_ = (_e725_ == 0u);
                                }
                                let _e729_ = phi_1270_;
                                local_1_ = array<f32, 4>(bitcast<f32>(select(_e697_, 0u, _e702_)), bitcast<f32>(select(_e706_, 0u, _e711_)), bitcast<f32>(select(_e715_, 0u, _e720_)), bitcast<f32>(select(_e724_, 0u, _e729_)));
                                let _e733_ = (_e645_ + 2u);
                                let _e734_ = (_e733_ < _e166_);
                                if _e734_ {
                                    if _e734_ {
                                    } else {
                                        break;
                                    }
                                    let _e737_ = tile_checkpoints.member[_e733_];
                                    phi_1294_ = _e737_;
                                } else {
                                    phi_1294_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e739_ = phi_1294_;
                                let _e741_ = bitcast<u32>(_e739_.x);
                                let _e742_ = (_e741_ & 2139095040u);
                                if (_e742_ == 2139095040u) {
                                    phi_1306_ = true;
                                } else {
                                    phi_1306_ = (_e742_ == 0u);
                                }
                                let _e746_ = phi_1306_;
                                let _e750_ = bitcast<u32>(_e739_.y);
                                let _e751_ = (_e750_ & 2139095040u);
                                if (_e751_ == 2139095040u) {
                                    phi_1323_ = true;
                                } else {
                                    phi_1323_ = (_e751_ == 0u);
                                }
                                let _e755_ = phi_1323_;
                                let _e759_ = bitcast<u32>(_e739_.z);
                                let _e760_ = (_e759_ & 2139095040u);
                                if (_e760_ == 2139095040u) {
                                    phi_1340_ = true;
                                } else {
                                    phi_1340_ = (_e760_ == 0u);
                                }
                                let _e764_ = phi_1340_;
                                let _e768_ = bitcast<u32>(_e739_.w);
                                let _e769_ = (_e768_ & 2139095040u);
                                if (_e769_ == 2139095040u) {
                                    phi_1357_ = true;
                                } else {
                                    phi_1357_ = (_e769_ == 0u);
                                }
                                let _e773_ = phi_1357_;
                                local_2_ = array<f32, 4>(bitcast<f32>(select(_e741_, 0u, _e746_)), bitcast<f32>(select(_e750_, 0u, _e755_)), bitcast<f32>(select(_e759_, 0u, _e764_)), bitcast<f32>(select(_e768_, 0u, _e773_)));
                                let _e777_ = (_e645_ + 3u);
                                let _e778_ = (_e777_ < _e166_);
                                if _e778_ {
                                    if _e778_ {
                                    } else {
                                        break;
                                    }
                                    let _e781_ = tile_checkpoints.member[_e777_];
                                    phi_1381_ = _e781_;
                                } else {
                                    phi_1381_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e783_ = phi_1381_;
                                let _e785_ = bitcast<u32>(_e783_.x);
                                let _e786_ = (_e785_ & 2139095040u);
                                if (_e786_ == 2139095040u) {
                                    phi_1393_ = true;
                                } else {
                                    phi_1393_ = (_e786_ == 0u);
                                }
                                let _e790_ = phi_1393_;
                                let _e794_ = bitcast<u32>(_e783_.y);
                                let _e795_ = (_e794_ & 2139095040u);
                                if (_e795_ == 2139095040u) {
                                    phi_1410_ = true;
                                } else {
                                    phi_1410_ = (_e795_ == 0u);
                                }
                                let _e799_ = phi_1410_;
                                let _e803_ = bitcast<u32>(_e783_.z);
                                let _e804_ = (_e803_ & 2139095040u);
                                if (_e804_ == 2139095040u) {
                                    phi_1427_ = true;
                                } else {
                                    phi_1427_ = (_e804_ == 0u);
                                }
                                let _e808_ = phi_1427_;
                                let _e812_ = bitcast<u32>(_e783_.w);
                                let _e813_ = (_e812_ & 2139095040u);
                                if (_e813_ == 2139095040u) {
                                    phi_1444_ = true;
                                } else {
                                    phi_1444_ = (_e813_ == 0u);
                                }
                                let _e817_ = phi_1444_;
                                local_3_ = array<f32, 4>(bitcast<f32>(select(_e785_, 0u, _e790_)), bitcast<f32>(select(_e794_, 0u, _e799_)), bitcast<f32>(select(_e803_, 0u, _e808_)), bitcast<f32>(select(_e812_, 0u, _e817_)));
                                let _e821_ = (_e645_ + 4u);
                                let _e822_ = (_e821_ < _e166_);
                                if _e822_ {
                                    if _e822_ {
                                    } else {
                                        break;
                                    }
                                    let _e825_ = tile_checkpoints.member[_e821_];
                                    phi_1468_ = _e825_;
                                } else {
                                    phi_1468_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e827_ = phi_1468_;
                                let _e829_ = bitcast<u32>(_e827_.x);
                                let _e830_ = (_e829_ & 2139095040u);
                                if (_e830_ == 2139095040u) {
                                    phi_1480_ = true;
                                } else {
                                    phi_1480_ = (_e830_ == 0u);
                                }
                                let _e834_ = phi_1480_;
                                let _e838_ = bitcast<u32>(_e827_.y);
                                let _e839_ = (_e838_ & 2139095040u);
                                if (_e839_ == 2139095040u) {
                                    phi_1497_ = true;
                                } else {
                                    phi_1497_ = (_e839_ == 0u);
                                }
                                let _e843_ = phi_1497_;
                                let _e847_ = bitcast<u32>(_e827_.z);
                                let _e848_ = (_e847_ & 2139095040u);
                                if (_e848_ == 2139095040u) {
                                    phi_1514_ = true;
                                } else {
                                    phi_1514_ = (_e848_ == 0u);
                                }
                                let _e852_ = phi_1514_;
                                let _e856_ = bitcast<u32>(_e827_.w);
                                let _e857_ = (_e856_ & 2139095040u);
                                if (_e857_ == 2139095040u) {
                                    phi_1531_ = true;
                                } else {
                                    phi_1531_ = (_e857_ == 0u);
                                }
                                let _e861_ = phi_1531_;
                                local_4_ = array<f32, 4>(bitcast<f32>(select(_e829_, 0u, _e834_)), bitcast<f32>(select(_e838_, 0u, _e843_)), bitcast<f32>(select(_e847_, 0u, _e852_)), bitcast<f32>(select(_e856_, 0u, _e861_)));
                                let _e865_ = (_e645_ + 5u);
                                let _e866_ = (_e865_ < _e166_);
                                if _e866_ {
                                    if _e866_ {
                                    } else {
                                        break;
                                    }
                                    let _e869_ = tile_checkpoints.member[_e865_];
                                    phi_1555_ = _e869_;
                                } else {
                                    phi_1555_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e871_ = phi_1555_;
                                let _e873_ = bitcast<u32>(_e871_.x);
                                let _e874_ = (_e873_ & 2139095040u);
                                if (_e874_ == 2139095040u) {
                                    phi_1567_ = true;
                                } else {
                                    phi_1567_ = (_e874_ == 0u);
                                }
                                let _e878_ = phi_1567_;
                                let _e882_ = bitcast<u32>(_e871_.y);
                                let _e883_ = (_e882_ & 2139095040u);
                                if (_e883_ == 2139095040u) {
                                    phi_1584_ = true;
                                } else {
                                    phi_1584_ = (_e883_ == 0u);
                                }
                                let _e887_ = phi_1584_;
                                let _e891_ = bitcast<u32>(_e871_.z);
                                let _e892_ = (_e891_ & 2139095040u);
                                if (_e892_ == 2139095040u) {
                                    phi_1601_ = true;
                                } else {
                                    phi_1601_ = (_e892_ == 0u);
                                }
                                let _e896_ = phi_1601_;
                                let _e900_ = bitcast<u32>(_e871_.w);
                                let _e901_ = (_e900_ & 2139095040u);
                                if (_e901_ == 2139095040u) {
                                    phi_1618_ = true;
                                } else {
                                    phi_1618_ = (_e901_ == 0u);
                                }
                                let _e905_ = phi_1618_;
                                local_5_ = array<f32, 4>(bitcast<f32>(select(_e873_, 0u, _e878_)), bitcast<f32>(select(_e882_, 0u, _e887_)), bitcast<f32>(select(_e891_, 0u, _e896_)), bitcast<f32>(select(_e900_, 0u, _e905_)));
                                let _e909_ = (_e645_ + 6u);
                                let _e910_ = (_e909_ < _e166_);
                                if _e910_ {
                                    if _e910_ {
                                    } else {
                                        break;
                                    }
                                    let _e913_ = tile_checkpoints.member[_e909_];
                                    phi_1642_ = _e913_;
                                } else {
                                    phi_1642_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e915_ = phi_1642_;
                                let _e917_ = bitcast<u32>(_e915_.x);
                                let _e918_ = (_e917_ & 2139095040u);
                                if (_e918_ == 2139095040u) {
                                    phi_1654_ = true;
                                } else {
                                    phi_1654_ = (_e918_ == 0u);
                                }
                                let _e922_ = phi_1654_;
                                let _e924_ = bitcast<f32>(select(_e917_, 0u, _e922_));
                                if (_e924_ < -1000000000000000000f) {
                                    phi_1671_ = -1000000000000000000f;
                                } else {
                                    phi_1671_ = select(_e924_, 1000000000000000000f, (_e924_ > 1000000000000000000f));
                                }
                                let _e929_ = phi_1671_;
                                let _e931_ = bitcast<u32>(_e915_.y);
                                let _e932_ = (_e931_ & 2139095040u);
                                if (_e932_ == 2139095040u) {
                                    phi_1683_ = true;
                                } else {
                                    phi_1683_ = (_e932_ == 0u);
                                }
                                let _e936_ = phi_1683_;
                                let _e938_ = bitcast<f32>(select(_e931_, 0u, _e936_));
                                if (_e938_ < -1000000000000000000f) {
                                    phi_1700_ = -1000000000000000000f;
                                } else {
                                    phi_1700_ = select(_e938_, 1000000000000000000f, (_e938_ > 1000000000000000000f));
                                }
                                let _e943_ = phi_1700_;
                                let _e945_ = bitcast<u32>(_e915_.z);
                                let _e946_ = (_e945_ & 2139095040u);
                                if (_e946_ == 2139095040u) {
                                    phi_1712_ = true;
                                } else {
                                    phi_1712_ = (_e946_ == 0u);
                                }
                                let _e950_ = phi_1712_;
                                let _e952_ = bitcast<f32>(select(_e945_, 0u, _e950_));
                                if (_e952_ < -1000000000000000000f) {
                                    phi_1729_ = -1000000000000000000f;
                                } else {
                                    phi_1729_ = select(_e952_, 1000000000000000000f, (_e952_ > 1000000000000000000f));
                                }
                                let _e957_ = phi_1729_;
                                let _e959_ = bitcast<u32>(_e915_.w);
                                let _e960_ = (_e959_ & 2139095040u);
                                if (_e960_ == 2139095040u) {
                                    phi_1741_ = true;
                                } else {
                                    phi_1741_ = (_e960_ == 0u);
                                }
                                let _e964_ = phi_1741_;
                                let _e966_ = bitcast<f32>(select(_e959_, 0u, _e964_));
                                if (_e966_ < -1000000000000000000f) {
                                    phi_1758_ = -1000000000000000000f;
                                } else {
                                    phi_1758_ = select(_e966_, 1000000000000000000f, (_e966_ > 1000000000000000000f));
                                }
                                let _e971_ = phi_1758_;
                                local_6_ = array<f32, 4>(_e929_, _e943_, _e957_, _e971_);
                                let _e973_ = (_e645_ + 7u);
                                let _e974_ = (_e973_ < _e166_);
                                if _e974_ {
                                    if _e974_ {
                                    } else {
                                        break;
                                    }
                                    let _e977_ = tile_checkpoints.member[_e973_];
                                    phi_1777_ = _e977_;
                                } else {
                                    phi_1777_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e979_ = phi_1777_;
                                let _e981_ = bitcast<u32>(_e979_.x);
                                let _e982_ = (_e981_ & 2139095040u);
                                if (_e982_ == 2139095040u) {
                                    phi_1789_ = true;
                                } else {
                                    phi_1789_ = (_e982_ == 0u);
                                }
                                let _e986_ = phi_1789_;
                                let _e988_ = bitcast<f32>(select(_e981_, 0u, _e986_));
                                if (_e988_ < -1000000000000000000f) {
                                    phi_1806_ = -1000000000000000000f;
                                } else {
                                    phi_1806_ = select(_e988_, 1000000000000000000f, (_e988_ > 1000000000000000000f));
                                }
                                let _e993_ = phi_1806_;
                                let _e995_ = bitcast<u32>(_e979_.y);
                                let _e996_ = (_e995_ & 2139095040u);
                                if (_e996_ == 2139095040u) {
                                    phi_1818_ = true;
                                } else {
                                    phi_1818_ = (_e996_ == 0u);
                                }
                                let _e1000_ = phi_1818_;
                                let _e1002_ = bitcast<f32>(select(_e995_, 0u, _e1000_));
                                if (_e1002_ < -1000000000000000000f) {
                                    phi_1835_ = -1000000000000000000f;
                                } else {
                                    phi_1835_ = select(_e1002_, 1000000000000000000f, (_e1002_ > 1000000000000000000f));
                                }
                                let _e1007_ = phi_1835_;
                                let _e1009_ = bitcast<u32>(_e979_.z);
                                let _e1010_ = (_e1009_ & 2139095040u);
                                if (_e1010_ == 2139095040u) {
                                    phi_1847_ = true;
                                } else {
                                    phi_1847_ = (_e1010_ == 0u);
                                }
                                let _e1014_ = phi_1847_;
                                let _e1016_ = bitcast<f32>(select(_e1009_, 0u, _e1014_));
                                if (_e1016_ < -1000000000000000000f) {
                                    phi_1864_ = -1000000000000000000f;
                                } else {
                                    phi_1864_ = select(_e1016_, 1000000000000000000f, (_e1016_ > 1000000000000000000f));
                                }
                                let _e1021_ = phi_1864_;
                                let _e1023_ = bitcast<u32>(_e979_.w);
                                let _e1024_ = (_e1023_ & 2139095040u);
                                if (_e1024_ == 2139095040u) {
                                    phi_1876_ = true;
                                } else {
                                    phi_1876_ = (_e1024_ == 0u);
                                }
                                let _e1028_ = phi_1876_;
                                let _e1030_ = bitcast<f32>(select(_e1023_, 0u, _e1028_));
                                if (_e1030_ < -1000000000000000000f) {
                                    phi_1893_ = -1000000000000000000f;
                                } else {
                                    phi_1893_ = select(_e1030_, 1000000000000000000f, (_e1030_ > 1000000000000000000f));
                                }
                                let _e1035_ = phi_1893_;
                                local_7_ = array<f32, 4>(_e993_, _e1007_, _e1021_, _e1035_);
                                phi_1902_ = type_15_(0u, 4u);
                                loop {
                                    let _e1038_ = phi_1902_;
                                    if (_e1038_.member < _e1038_.member_1_) {
                                        phi_1903_ = type_15_((_e1038_.member + 1u), _e1038_.member_1_);
                                        phi_1926_ = type_15_(1u, _e1038_.member);
                                    } else {
                                        phi_1903_ = _e1038_;
                                        phi_1926_ = type_15_(0u, u32());
                                    }
                                    let _e1051_ = phi_1903_;
                                    let _e1053_ = phi_1926_;
                                    let _e1057_ = (bitcast<i32>(_e1053_.member) != 0i);
                                    if _e1057_ {
                                        let _e1058_ = (_e1053_.member_1_ < 4u);
                                        if _e1058_ {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        let _e1060_ = local[_e1053_.member_1_];
                                        if (_e1053_.member_1_ < 256u) {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1053_.member_1_] = _e1060_;
                                        if _e1058_ {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        let _e1064_ = local_2_[_e1053_.member_1_];
                                        let _e1065_ = (4u + _e1053_.member_1_);
                                        if (_e1065_ < 256u) {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1065_] = _e1064_;
                                        if _e1058_ {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        let _e1069_ = local_1_[_e1053_.member_1_];
                                        let _e1070_ = (8u + _e1053_.member_1_);
                                        if (_e1070_ < 256u) {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1070_] = _e1069_;
                                        if _e1058_ {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        let _e1074_ = local_3_[_e1053_.member_1_];
                                        let _e1075_ = (12u + _e1053_.member_1_);
                                        if (_e1075_ < 256u) {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1075_] = _e1074_;
                                        if _e1058_ {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        let _e1079_ = local_4_[_e1053_.member_1_];
                                        let _e1080_ = (16u + _e1053_.member_1_);
                                        if (_e1080_ < 256u) {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1080_] = _e1079_;
                                        if _e1058_ {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        let _e1084_ = local_5_[_e1053_.member_1_];
                                        let _e1085_ = (20u + _e1053_.member_1_);
                                        if (_e1085_ < 256u) {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1085_] = _e1084_;
                                        if _e1058_ {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        let _e1089_ = local_6_[_e1053_.member_1_];
                                        let _e1090_ = (24u + _e1053_.member_1_);
                                        if (_e1090_ < 256u) {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1090_] = _e1089_;
                                        if _e1058_ {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        let _e1094_ = local_7_[_e1053_.member_1_];
                                        let _e1095_ = (28u + _e1053_.member_1_);
                                        if (_e1095_ < 256u) {
                                        } else {
                                            phi_33936_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1095_] = _e1094_;
                                    }
                                    phi_1902_ = _e1051_;
                                    phi_33936_ = false;
                                    if !(_e1057_) {
                                        break;
                                    }
                                }
                                let _e1100_ = phi_33936_;
                                if _e1100_ {
                                    break;
                                }
                                phi_34132_ = _e1100_;
                            } else {
                                phi_34132_ = false;
                            }
                            let _e1102_ = phi_34132_;
                            phi_34131_ = _e1102_;
                            phi_2046_ = select(true, false, _e641_);
                        } else {
                            phi_34131_ = false;
                            phi_2046_ = false;
                        }
                        let _e1105_ = phi_34131_;
                        let _e1107_ = phi_2046_;
                        phi_34130_ = _e1105_;
                        phi_2048_ = _e641_;
                        phi_2052_ = _e1107_;
                        phi_2053_ = select(false, _e641_, _e642_);
                    } else {
                        phi_34130_ = false;
                        phi_2048_ = bool();
                        phi_2052_ = false;
                        phi_2053_ = false;
                    }
                    let _e1110_ = phi_34130_;
                    let _e1112_ = phi_2048_;
                    let _e1114_ = phi_2052_;
                    let _e1116_ = phi_2053_;
                    phi_34129_ = _e1110_;
                    phi_2054_ = _e1112_;
                    phi_2057_ = select(true, false, _e640_);
                    phi_2058_ = _e1114_;
                    phi_2059_ = _e1116_;
                } else {
                    phi_34129_ = false;
                    phi_2054_ = bool();
                    phi_2057_ = true;
                    phi_2058_ = false;
                    phi_2059_ = false;
                }
                let _e1119_ = phi_34129_;
                let _e1121_ = phi_2054_;
                let _e1123_ = phi_2057_;
                let _e1125_ = phi_2058_;
                let _e1127_ = phi_2059_;
                if _e1123_ {
                    phi_2069_ = (_e169_ == 0u);
                } else {
                    phi_2069_ = _e1125_;
                }
                let _e1130_ = phi_2069_;
                let _e1131_ = select(_e1121_, false, _e1123_);
                if _e1130_ {
                    phi_2076_ = type_15_(0u, 32u);
                    loop {
                        let _e1134_ = phi_2076_;
                        if (_e1134_.member < _e1134_.member_1_) {
                            phi_2077_ = type_15_((_e1134_.member + 1u), _e1134_.member_1_);
                            phi_2100_ = type_15_(1u, _e1134_.member);
                        } else {
                            phi_2077_ = _e1134_;
                            phi_2100_ = type_15_(0u, u32());
                        }
                        let _e1147_ = phi_2077_;
                        let _e1149_ = phi_2100_;
                        let _e1153_ = (bitcast<i32>(_e1149_.member) != 0i);
                        if _e1153_ {
                            if (_e1149_.member_1_ < 256u) {
                            } else {
                                phi_34124_ = true;
                                break;
                            }
                            aux_scratch[_e1149_.member_1_] = 0f;
                        }
                        phi_2076_ = _e1147_;
                        phi_34124_ = _e1119_;
                        if !(_e1153_) {
                            break;
                        }
                    }
                    let _e1158_ = phi_34124_;
                    if _e1158_ {
                        break;
                    }
                    phi_34165_ = _e1158_;
                } else {
                    phi_34165_ = _e1119_;
                }
                let _e1160_ = phi_34165_;
                workgroupBarrier();
                phi_34162_ = _e1160_;
                phi_2126_ = type_15_(_e303_, _e305_);
                loop {
                    let _e1167_ = phi_34162_;
                    let _e1169_ = phi_2126_;
                    if (_e1169_.member < _e1169_.member_1_) {
                        phi_2127_ = type_15_((_e1169_.member + 1u), _e1169_.member_1_);
                        phi_2150_ = type_15_(1u, _e1169_.member);
                    } else {
                        phi_2127_ = _e1169_;
                        phi_2150_ = type_15_(0u, u32());
                    }
                    let _e1182_ = phi_2127_;
                    let _e1184_ = phi_2150_;
                    let _e1188_ = (bitcast<i32>(_e1184_.member) != 0i);
                    if _e1188_ {
                        let _e1190_ = (_e169_ == 0u);
                        if _e1190_ {
                            if select(select(_e1121_, _e1131_, _e1130_), _e1131_, select(true, false, select(select(_e1127_, false, _e1123_), true, _e1130_))) {
                                phi_2171_ = select(true, false, (_e1184_.member_1_ == _e235_));
                            } else {
                                phi_2171_ = true;
                            }
                            let _e1194_ = phi_2171_;
                            if _e1194_ {
                                let _e1197_ = (((_e205_ * _e228_) + _e1184_.member_1_) * 8u);
                                let _e1199_ = aux_scratch[0];
                                let _e1201_ = aux_scratch[1];
                                let _e1203_ = aux_scratch[2];
                                let _e1205_ = aux_scratch[3];
                                let _e1207_ = aux_scratch[4];
                                let _e1209_ = aux_scratch[5];
                                let _e1211_ = aux_scratch[6];
                                let _e1213_ = aux_scratch[7];
                                let _e1215_ = aux_scratch[8];
                                let _e1217_ = aux_scratch[9];
                                let _e1219_ = aux_scratch[10];
                                let _e1221_ = aux_scratch[11];
                                let _e1223_ = aux_scratch[12];
                                let _e1225_ = aux_scratch[13];
                                let _e1227_ = aux_scratch[14];
                                let _e1229_ = aux_scratch[15];
                                let _e1231_ = aux_scratch[16];
                                let _e1233_ = aux_scratch[17];
                                let _e1235_ = aux_scratch[18];
                                let _e1237_ = aux_scratch[19];
                                let _e1239_ = aux_scratch[20];
                                let _e1241_ = aux_scratch[21];
                                let _e1243_ = aux_scratch[22];
                                let _e1245_ = aux_scratch[23];
                                let _e1247_ = aux_scratch[24];
                                let _e1249_ = aux_scratch[25];
                                let _e1251_ = aux_scratch[26];
                                let _e1253_ = aux_scratch[27];
                                let _e1255_ = aux_scratch[28];
                                let _e1257_ = aux_scratch[29];
                                let _e1259_ = aux_scratch[30];
                                let _e1261_ = aux_scratch[31];
                                let _e1262_ = bitcast<u32>(_e1199_);
                                let _e1263_ = (_e1262_ & 2139095040u);
                                if (_e1263_ == 2139095040u) {
                                    phi_2252_ = true;
                                } else {
                                    phi_2252_ = (_e1263_ == 0u);
                                }
                                let _e1267_ = phi_2252_;
                                let _e1270_ = bitcast<u32>(_e1201_);
                                let _e1271_ = (_e1270_ & 2139095040u);
                                if (_e1271_ == 2139095040u) {
                                    phi_2268_ = true;
                                } else {
                                    phi_2268_ = (_e1271_ == 0u);
                                }
                                let _e1275_ = phi_2268_;
                                let _e1278_ = bitcast<u32>(_e1203_);
                                let _e1279_ = (_e1278_ & 2139095040u);
                                if (_e1279_ == 2139095040u) {
                                    phi_2284_ = true;
                                } else {
                                    phi_2284_ = (_e1279_ == 0u);
                                }
                                let _e1283_ = phi_2284_;
                                let _e1286_ = bitcast<u32>(_e1205_);
                                let _e1287_ = (_e1286_ & 2139095040u);
                                if (_e1287_ == 2139095040u) {
                                    phi_2300_ = true;
                                } else {
                                    phi_2300_ = (_e1287_ == 0u);
                                }
                                let _e1291_ = phi_2300_;
                                let _e1295_ = (_e1197_ < _e166_);
                                if _e1295_ {
                                    if _e1295_ {
                                    } else {
                                        phi_36582_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1197_] = vec4<f32>(bitcast<f32>(select(_e1262_, 0u, _e1267_)), bitcast<f32>(select(_e1270_, 0u, _e1275_)), bitcast<f32>(select(_e1278_, 0u, _e1283_)), bitcast<f32>(select(_e1286_, 0u, _e1291_)));
                                }
                                let _e1298_ = (_e1197_ + 1u);
                                let _e1299_ = bitcast<u32>(_e1215_);
                                let _e1300_ = (_e1299_ & 2139095040u);
                                if (_e1300_ == 2139095040u) {
                                    phi_2328_ = true;
                                } else {
                                    phi_2328_ = (_e1300_ == 0u);
                                }
                                let _e1304_ = phi_2328_;
                                let _e1307_ = bitcast<u32>(_e1217_);
                                let _e1308_ = (_e1307_ & 2139095040u);
                                if (_e1308_ == 2139095040u) {
                                    phi_2344_ = true;
                                } else {
                                    phi_2344_ = (_e1308_ == 0u);
                                }
                                let _e1312_ = phi_2344_;
                                let _e1315_ = bitcast<u32>(_e1219_);
                                let _e1316_ = (_e1315_ & 2139095040u);
                                if (_e1316_ == 2139095040u) {
                                    phi_2360_ = true;
                                } else {
                                    phi_2360_ = (_e1316_ == 0u);
                                }
                                let _e1320_ = phi_2360_;
                                let _e1323_ = bitcast<u32>(_e1221_);
                                let _e1324_ = (_e1323_ & 2139095040u);
                                if (_e1324_ == 2139095040u) {
                                    phi_2376_ = true;
                                } else {
                                    phi_2376_ = (_e1324_ == 0u);
                                }
                                let _e1328_ = phi_2376_;
                                let _e1332_ = (_e1298_ < _e166_);
                                if _e1332_ {
                                    if _e1332_ {
                                    } else {
                                        phi_36582_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1298_] = vec4<f32>(bitcast<f32>(select(_e1299_, 0u, _e1304_)), bitcast<f32>(select(_e1307_, 0u, _e1312_)), bitcast<f32>(select(_e1315_, 0u, _e1320_)), bitcast<f32>(select(_e1323_, 0u, _e1328_)));
                                }
                                let _e1335_ = (_e1197_ + 2u);
                                let _e1336_ = bitcast<u32>(_e1207_);
                                let _e1337_ = (_e1336_ & 2139095040u);
                                if (_e1337_ == 2139095040u) {
                                    phi_2404_ = true;
                                } else {
                                    phi_2404_ = (_e1337_ == 0u);
                                }
                                let _e1341_ = phi_2404_;
                                let _e1344_ = bitcast<u32>(_e1209_);
                                let _e1345_ = (_e1344_ & 2139095040u);
                                if (_e1345_ == 2139095040u) {
                                    phi_2420_ = true;
                                } else {
                                    phi_2420_ = (_e1345_ == 0u);
                                }
                                let _e1349_ = phi_2420_;
                                let _e1352_ = bitcast<u32>(_e1211_);
                                let _e1353_ = (_e1352_ & 2139095040u);
                                if (_e1353_ == 2139095040u) {
                                    phi_2436_ = true;
                                } else {
                                    phi_2436_ = (_e1353_ == 0u);
                                }
                                let _e1357_ = phi_2436_;
                                let _e1360_ = bitcast<u32>(_e1213_);
                                let _e1361_ = (_e1360_ & 2139095040u);
                                if (_e1361_ == 2139095040u) {
                                    phi_2452_ = true;
                                } else {
                                    phi_2452_ = (_e1361_ == 0u);
                                }
                                let _e1365_ = phi_2452_;
                                let _e1369_ = (_e1335_ < _e166_);
                                if _e1369_ {
                                    if _e1369_ {
                                    } else {
                                        phi_36582_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1335_] = vec4<f32>(bitcast<f32>(select(_e1336_, 0u, _e1341_)), bitcast<f32>(select(_e1344_, 0u, _e1349_)), bitcast<f32>(select(_e1352_, 0u, _e1357_)), bitcast<f32>(select(_e1360_, 0u, _e1365_)));
                                }
                                let _e1372_ = (_e1197_ + 3u);
                                let _e1373_ = bitcast<u32>(_e1223_);
                                let _e1374_ = (_e1373_ & 2139095040u);
                                if (_e1374_ == 2139095040u) {
                                    phi_2480_ = true;
                                } else {
                                    phi_2480_ = (_e1374_ == 0u);
                                }
                                let _e1378_ = phi_2480_;
                                let _e1381_ = bitcast<u32>(_e1225_);
                                let _e1382_ = (_e1381_ & 2139095040u);
                                if (_e1382_ == 2139095040u) {
                                    phi_2496_ = true;
                                } else {
                                    phi_2496_ = (_e1382_ == 0u);
                                }
                                let _e1386_ = phi_2496_;
                                let _e1389_ = bitcast<u32>(_e1227_);
                                let _e1390_ = (_e1389_ & 2139095040u);
                                if (_e1390_ == 2139095040u) {
                                    phi_2512_ = true;
                                } else {
                                    phi_2512_ = (_e1390_ == 0u);
                                }
                                let _e1394_ = phi_2512_;
                                let _e1397_ = bitcast<u32>(_e1229_);
                                let _e1398_ = (_e1397_ & 2139095040u);
                                if (_e1398_ == 2139095040u) {
                                    phi_2528_ = true;
                                } else {
                                    phi_2528_ = (_e1398_ == 0u);
                                }
                                let _e1402_ = phi_2528_;
                                let _e1406_ = (_e1372_ < _e166_);
                                if _e1406_ {
                                    if _e1406_ {
                                    } else {
                                        phi_36582_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1372_] = vec4<f32>(bitcast<f32>(select(_e1373_, 0u, _e1378_)), bitcast<f32>(select(_e1381_, 0u, _e1386_)), bitcast<f32>(select(_e1389_, 0u, _e1394_)), bitcast<f32>(select(_e1397_, 0u, _e1402_)));
                                }
                                let _e1409_ = (_e1197_ + 4u);
                                let _e1410_ = bitcast<u32>(_e1231_);
                                let _e1411_ = (_e1410_ & 2139095040u);
                                if (_e1411_ == 2139095040u) {
                                    phi_2556_ = true;
                                } else {
                                    phi_2556_ = (_e1411_ == 0u);
                                }
                                let _e1415_ = phi_2556_;
                                let _e1418_ = bitcast<u32>(_e1233_);
                                let _e1419_ = (_e1418_ & 2139095040u);
                                if (_e1419_ == 2139095040u) {
                                    phi_2572_ = true;
                                } else {
                                    phi_2572_ = (_e1419_ == 0u);
                                }
                                let _e1423_ = phi_2572_;
                                let _e1426_ = bitcast<u32>(_e1235_);
                                let _e1427_ = (_e1426_ & 2139095040u);
                                if (_e1427_ == 2139095040u) {
                                    phi_2588_ = true;
                                } else {
                                    phi_2588_ = (_e1427_ == 0u);
                                }
                                let _e1431_ = phi_2588_;
                                let _e1434_ = bitcast<u32>(_e1237_);
                                let _e1435_ = (_e1434_ & 2139095040u);
                                if (_e1435_ == 2139095040u) {
                                    phi_2604_ = true;
                                } else {
                                    phi_2604_ = (_e1435_ == 0u);
                                }
                                let _e1439_ = phi_2604_;
                                let _e1443_ = (_e1409_ < _e166_);
                                if _e1443_ {
                                    if _e1443_ {
                                    } else {
                                        phi_36582_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1409_] = vec4<f32>(bitcast<f32>(select(_e1410_, 0u, _e1415_)), bitcast<f32>(select(_e1418_, 0u, _e1423_)), bitcast<f32>(select(_e1426_, 0u, _e1431_)), bitcast<f32>(select(_e1434_, 0u, _e1439_)));
                                }
                                let _e1446_ = (_e1197_ + 5u);
                                let _e1447_ = bitcast<u32>(_e1239_);
                                let _e1448_ = (_e1447_ & 2139095040u);
                                if (_e1448_ == 2139095040u) {
                                    phi_2632_ = true;
                                } else {
                                    phi_2632_ = (_e1448_ == 0u);
                                }
                                let _e1452_ = phi_2632_;
                                let _e1455_ = bitcast<u32>(_e1241_);
                                let _e1456_ = (_e1455_ & 2139095040u);
                                if (_e1456_ == 2139095040u) {
                                    phi_2648_ = true;
                                } else {
                                    phi_2648_ = (_e1456_ == 0u);
                                }
                                let _e1460_ = phi_2648_;
                                let _e1463_ = bitcast<u32>(_e1243_);
                                let _e1464_ = (_e1463_ & 2139095040u);
                                if (_e1464_ == 2139095040u) {
                                    phi_2664_ = true;
                                } else {
                                    phi_2664_ = (_e1464_ == 0u);
                                }
                                let _e1468_ = phi_2664_;
                                let _e1471_ = bitcast<u32>(_e1245_);
                                let _e1472_ = (_e1471_ & 2139095040u);
                                if (_e1472_ == 2139095040u) {
                                    phi_2680_ = true;
                                } else {
                                    phi_2680_ = (_e1472_ == 0u);
                                }
                                let _e1476_ = phi_2680_;
                                let _e1480_ = (_e1446_ < _e166_);
                                if _e1480_ {
                                    if _e1480_ {
                                    } else {
                                        phi_36582_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1446_] = vec4<f32>(bitcast<f32>(select(_e1447_, 0u, _e1452_)), bitcast<f32>(select(_e1455_, 0u, _e1460_)), bitcast<f32>(select(_e1463_, 0u, _e1468_)), bitcast<f32>(select(_e1471_, 0u, _e1476_)));
                                }
                                let _e1483_ = (_e1197_ + 6u);
                                let _e1484_ = bitcast<u32>(_e1247_);
                                let _e1485_ = (_e1484_ & 2139095040u);
                                if (_e1485_ == 2139095040u) {
                                    phi_2708_ = true;
                                } else {
                                    phi_2708_ = (_e1485_ == 0u);
                                }
                                let _e1489_ = phi_2708_;
                                let _e1492_ = bitcast<u32>(_e1249_);
                                let _e1493_ = (_e1492_ & 2139095040u);
                                if (_e1493_ == 2139095040u) {
                                    phi_2724_ = true;
                                } else {
                                    phi_2724_ = (_e1493_ == 0u);
                                }
                                let _e1497_ = phi_2724_;
                                let _e1500_ = bitcast<u32>(_e1251_);
                                let _e1501_ = (_e1500_ & 2139095040u);
                                if (_e1501_ == 2139095040u) {
                                    phi_2740_ = true;
                                } else {
                                    phi_2740_ = (_e1501_ == 0u);
                                }
                                let _e1505_ = phi_2740_;
                                let _e1508_ = bitcast<u32>(_e1253_);
                                let _e1509_ = (_e1508_ & 2139095040u);
                                if (_e1509_ == 2139095040u) {
                                    phi_2756_ = true;
                                } else {
                                    phi_2756_ = (_e1509_ == 0u);
                                }
                                let _e1513_ = phi_2756_;
                                let _e1517_ = (_e1483_ < _e166_);
                                if _e1517_ {
                                    if _e1517_ {
                                    } else {
                                        phi_36582_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1483_] = vec4<f32>(bitcast<f32>(select(_e1484_, 0u, _e1489_)), bitcast<f32>(select(_e1492_, 0u, _e1497_)), bitcast<f32>(select(_e1500_, 0u, _e1505_)), bitcast<f32>(select(_e1508_, 0u, _e1513_)));
                                }
                                let _e1520_ = (_e1197_ + 7u);
                                let _e1521_ = bitcast<u32>(_e1255_);
                                let _e1522_ = (_e1521_ & 2139095040u);
                                if (_e1522_ == 2139095040u) {
                                    phi_2784_ = true;
                                } else {
                                    phi_2784_ = (_e1522_ == 0u);
                                }
                                let _e1526_ = phi_2784_;
                                let _e1529_ = bitcast<u32>(_e1257_);
                                let _e1530_ = (_e1529_ & 2139095040u);
                                if (_e1530_ == 2139095040u) {
                                    phi_2800_ = true;
                                } else {
                                    phi_2800_ = (_e1530_ == 0u);
                                }
                                let _e1534_ = phi_2800_;
                                let _e1537_ = bitcast<u32>(_e1259_);
                                let _e1538_ = (_e1537_ & 2139095040u);
                                if (_e1538_ == 2139095040u) {
                                    phi_2816_ = true;
                                } else {
                                    phi_2816_ = (_e1538_ == 0u);
                                }
                                let _e1542_ = phi_2816_;
                                let _e1545_ = bitcast<u32>(_e1261_);
                                let _e1546_ = (_e1545_ & 2139095040u);
                                if (_e1546_ == 2139095040u) {
                                    phi_2832_ = true;
                                } else {
                                    phi_2832_ = (_e1546_ == 0u);
                                }
                                let _e1550_ = phi_2832_;
                                let _e1554_ = (_e1520_ < _e166_);
                                if _e1554_ {
                                    if _e1554_ {
                                    } else {
                                        phi_36582_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1520_] = vec4<f32>(bitcast<f32>(select(_e1521_, 0u, _e1526_)), bitcast<f32>(select(_e1529_, 0u, _e1534_)), bitcast<f32>(select(_e1537_, 0u, _e1542_)), bitcast<f32>(select(_e1545_, 0u, _e1550_)));
                                }
                            }
                        }
                        workgroupBarrier();
                        phi_34159_ = _e1167_;
                        phi_2851_ = type_15_(0u, 2u);
                        loop {
                            let _e1558_ = phi_34159_;
                            let _e1560_ = phi_2851_;
                            if (_e1560_.member < _e1560_.member_1_) {
                                phi_2852_ = type_15_((_e1560_.member + 1u), _e1560_.member_1_);
                                phi_2875_ = type_15_(1u, _e1560_.member);
                            } else {
                                phi_2852_ = _e1560_;
                                phi_2875_ = type_15_(0u, u32());
                            }
                            let _e1573_ = phi_2852_;
                            let _e1575_ = phi_2875_;
                            let _e1579_ = (bitcast<i32>(_e1575_.member) != 0i);
                            if _e1579_ {
                                let _e1581_ = ((_e1184_.member_1_ * 256u) + (_e1575_.member_1_ * 128u));
                                if (_e216_ > _e1581_) {
                                    phi_2890_ = (_e216_ - _e1581_);
                                } else {
                                    phi_2890_ = 0u;
                                }
                                let _e1585_ = phi_2890_;
                                let _e1587_ = select(_e1585_, 128u, (128u < _e1585_));
                                let _e1588_ = (_e1587_ > 0u);
                                if _e1588_ {
                                    let _e1589_ = (_e170_ < _e1587_);
                                    let _e1592_ = ((_e205_ * _e216_) + (_e1581_ + _e170_));
                                    let _e1593_ = (_e1592_ * 2u);
                                    if _e1589_ {
                                        let _e1594_ = (_e1593_ < _e162_);
                                        if _e1594_ {
                                            if _e1594_ {
                                            } else {
                                                phi_36576_ = true;
                                                break;
                                            }
                                            let _e1597_ = core_inputs.member[_e1593_];
                                            phi_2919_ = _e1597_;
                                        } else {
                                            phi_2919_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e1599_ = phi_2919_;
                                        let _e1600_ = (_e1593_ + 1u);
                                        let _e1601_ = (_e1600_ < _e162_);
                                        if _e1601_ {
                                            if _e1601_ {
                                            } else {
                                                phi_36576_ = true;
                                                break;
                                            }
                                            let _e1604_ = core_inputs.member[_e1600_];
                                            phi_2932_ = _e1604_;
                                        } else {
                                            phi_2932_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e1606_ = phi_2932_;
                                        phi_2933_ = _e1606_;
                                        phi_2934_ = _e1599_;
                                    } else {
                                        phi_2933_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        phi_2934_ = vec4<f32>(0f, 0f, 0f, 0f);
                                    }
                                    let _e1608_ = phi_2933_;
                                    let _e1610_ = phi_2934_;
                                    if _e213_ {
                                        if _e209_ {
                                            phi_34356_ = _e1558_;
                                            phi_2943_ = type_15_(0u, 4u);
                                            loop {
                                                let _e2266_ = phi_34356_;
                                                let _e2268_ = phi_2943_;
                                                if (_e2268_.member < _e2268_.member_1_) {
                                                    phi_2944_ = type_15_((_e2268_.member + 1u), _e2268_.member_1_);
                                                    phi_2967_ = type_15_(1u, _e2268_.member);
                                                } else {
                                                    phi_2944_ = _e2268_;
                                                    phi_2967_ = type_15_(0u, u32());
                                                }
                                                let _e2281_ = phi_2944_;
                                                let _e2283_ = phi_2967_;
                                                let _e2287_ = (bitcast<i32>(_e2283_.member) != 0i);
                                                if _e2287_ {
                                                    let _e2288_ = (_e2283_.member_1_ & 3u);
                                                    switch bitcast<i32>(_e2288_) {
                                                        case 0: {
                                                            phi_3080_ = _e333_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3080_ = _e333_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3080_ = _e333_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3080_ = _e333_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2295_ = phi_3080_;
                                                    switch bitcast<i32>(_e2288_) {
                                                        case 0: {
                                                            phi_3097_ = _e347_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3097_ = _e347_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3097_ = _e347_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3097_ = _e347_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2302_ = phi_3097_;
                                                    switch bitcast<i32>(_e2288_) {
                                                        case 0: {
                                                            phi_3108_ = _e354_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3108_ = _e354_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3108_ = _e354_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3108_ = _e354_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2309_ = phi_3108_;
                                                    switch bitcast<i32>(_e2288_) {
                                                        case 0: {
                                                            phi_3119_ = _e361_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3119_ = _e361_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3119_ = _e361_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3119_ = _e361_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2316_ = phi_3119_;
                                                    switch bitcast<i32>(_e2288_) {
                                                        case 0: {
                                                            phi_3130_ = _e368_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3130_ = _e368_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3130_ = _e368_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3130_ = _e368_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2323_ = phi_3130_;
                                                    switch bitcast<i32>(_e2288_) {
                                                        case 0: {
                                                            phi_3141_ = _e375_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3141_ = _e375_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3141_ = _e375_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3141_ = _e375_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2330_ = phi_3141_;
                                                    switch bitcast<i32>(_e2288_) {
                                                        case 0: {
                                                            phi_3356_ = _e424_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3356_ = _e424_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3356_ = _e424_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3356_ = _e424_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2337_ = phi_3356_;
                                                    switch bitcast<i32>(_e2288_) {
                                                        case 0: {
                                                            phi_3431_ = _e475_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_3431_ = _e475_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_3431_ = _e475_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_3431_ = _e475_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e2344_ = phi_3431_;
                                                    if _e1589_ {
                                                        let _e2346_ = select(_e1610_, _e1608_, vec4(_e173_));
                                                        switch bitcast<i32>(_e2288_) {
                                                            case 0: {
                                                                phi_3449_ = _e2346_.x;
                                                                break;
                                                            }
                                                            case 1: {
                                                                phi_3449_ = _e2346_.y;
                                                                break;
                                                            }
                                                            case 2: {
                                                                phi_3449_ = _e2346_.z;
                                                                break;
                                                            }
                                                            default: {
                                                                phi_3449_ = _e2346_.w;
                                                                break;
                                                            }
                                                        }
                                                        let _e2353_ = phi_3449_;
                                                        let _e2354_ = bitcast<u32>(_e2353_);
                                                        let _e2355_ = (_e2354_ & 2139095040u);
                                                        if (_e2355_ == 2139095040u) {
                                                            phi_3460_ = true;
                                                        } else {
                                                            phi_3460_ = (_e2355_ == 0u);
                                                        }
                                                        let _e2359_ = phi_3460_;
                                                        phi_3466_ = bitcast<f32>(select(_e2354_, 0u, _e2359_));
                                                    } else {
                                                        phi_3466_ = 0f;
                                                    }
                                                    let _e2363_ = phi_3466_;
                                                    if _e173_ {
                                                        phi_3475_ = (8u + _e2283_.member_1_);
                                                    } else {
                                                        phi_3475_ = _e2283_.member_1_;
                                                    }
                                                    let _e2366_ = phi_3475_;
                                                    if (_e2366_ < 256u) {
                                                    } else {
                                                        phi_34543_ = true;
                                                        break;
                                                    }
                                                    let _e2369_ = aux_scratch[_e2366_];
                                                    let _e2370_ = (_e2366_ + 4u);
                                                    if (_e2370_ < 256u) {
                                                    } else {
                                                        phi_34543_ = true;
                                                        break;
                                                    }
                                                    let _e2373_ = aux_scratch[_e2370_];
                                                    let _e2374_ = array<f32, 2>(_e2369_, _e2373_);
                                                    phi_34344_ = _e2266_;
                                                    phi_3494_ = type_15_(0u, 6u);
                                                    phi_3497_ = _e2374_;
                                                    phi_3499_ = _e2373_;
                                                    phi_3501_ = _e2369_;
                                                    phi_3503_ = _e2374_;
                                                    phi_3505_ = false;
                                                    loop {
                                                        let _e2376_ = phi_34344_;
                                                        let _e2378_ = phi_3494_;
                                                        let _e2380_ = phi_3497_;
                                                        let _e2382_ = phi_3499_;
                                                        let _e2384_ = phi_3501_;
                                                        let _e2386_ = phi_3503_;
                                                        let _e2388_ = phi_3505_;
                                                        local_24_ = _e2388_;
                                                        local_29_ = _e2384_;
                                                        local_30_ = _e2384_;
                                                        local_31_ = _e2382_;
                                                        local_32_ = _e2384_;
                                                        local_33_ = _e2384_;
                                                        local_34_ = _e2382_;
                                                        local_35_ = _e2386_;
                                                        local_36_ = _e2386_;
                                                        local_37_ = _e2386_;
                                                        local_38_ = _e2386_;
                                                        if (_e2378_.member < _e2378_.member_1_) {
                                                            phi_3495_ = type_15_((_e2378_.member + 1u), _e2378_.member_1_);
                                                            phi_3528_ = type_15_(1u, _e2378_.member);
                                                        } else {
                                                            phi_3495_ = _e2378_;
                                                            phi_3528_ = type_15_(0u, u32());
                                                        }
                                                        let _e2401_ = phi_3495_;
                                                        let _e2403_ = phi_3528_;
                                                        let _e2407_ = (bitcast<i32>(_e2403_.member) != 0i);
                                                        if _e2407_ {
                                                            if _e2388_ {
                                                                phi_34455_ = _e2376_;
                                                                phi_4625_ = _e2380_;
                                                                phi_4626_ = _e2382_;
                                                                phi_4627_ = _e2384_;
                                                                phi_4628_ = _e2386_;
                                                                phi_4629_ = _e2388_;
                                                            } else {
                                                                let _e2408_ = (_e2337_ * 0.5f);
                                                                if (_e2408_ != _e2408_) {
                                                                    phi_23956_ = true;
                                                                } else {
                                                                    phi_23956_ = (0.00001f >= _e2408_);
                                                                }
                                                                let _e2412_ = phi_23956_;
                                                                let _e2413_ = select(_e2408_, 0.00001f, _e2412_);
                                                                if (_e2295_ != _e2295_) {
                                                                    phi_23971_ = true;
                                                                } else {
                                                                    phi_23971_ = (0f >= _e2295_);
                                                                }
                                                                let _e2417_ = phi_23971_;
                                                                if (_e2344_ != _e2344_) {
                                                                    phi_23986_ = true;
                                                                } else {
                                                                    phi_23986_ = (0f >= _e2344_);
                                                                }
                                                                let _e2422_ = phi_23986_;
                                                                let _e2424_ = (_e2384_ * _e2384_);
                                                                if (_e2424_ != _e2424_) {
                                                                    phi_24001_ = true;
                                                                } else {
                                                                    phi_24001_ = (64f <= _e2424_);
                                                                }
                                                                let _e2428_ = phi_24001_;
                                                                let _e2434_ = (1f + (_e2413_ * (_e2413_ + (select(_e2295_, 0f, _e2417_) + (select(_e2344_, 0f, _e2422_) * select(_e2424_, 64f, _e2428_))))));
                                                                let _e2437_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2434_)));
                                                                let _e2441_ = (_e2437_ + (_e2437_ * (1f - (_e2434_ * _e2437_))));
                                                                let _e2445_ = (_e2441_ + (_e2441_ * (1f - (_e2434_ * _e2441_))));
                                                                let _e2446_ = (_e2413_ * _e2445_);
                                                                let _e2447_ = (_e2413_ * _e2446_);
                                                                if _e1589_ {
                                                                    let _e2448_ = (_e2446_ + _e2446_);
                                                                    let _e2449_ = (_e2447_ + _e2447_);
                                                                    phi_3576_ = type_32_(((_e2445_ + _e2445_) - 1f), -(_e2448_), _e2448_, (1f - _e2449_), (_e2448_ * _e2363_), (_e2449_ * _e2363_));
                                                                } else {
                                                                    phi_3576_ = type_32_(1f, 0f, 0f, 1f, 0f, 0f);
                                                                }
                                                                let _e2458_ = phi_3576_;
                                                                let _e2459_ = select(0u, 768u, _e173_);
                                                                let _e2460_ = (_e170_ & 127u);
                                                                let _e2462_ = (_e2459_ + _e2460_);
                                                                let _e2463_ = (_e2462_ < 1536u);
                                                                if _e2463_ {
                                                                } else {
                                                                    phi_34453_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2462_] = _e2458_.member;
                                                                let _e2466_ = (_e2459_ + 128u);
                                                                let _e2467_ = (_e2466_ + _e2460_);
                                                                let _e2468_ = (_e2467_ < 1536u);
                                                                if _e2468_ {
                                                                } else {
                                                                    phi_34453_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2467_] = _e2458_.member_1_;
                                                                let _e2471_ = (_e2459_ + 256u);
                                                                let _e2472_ = (_e2471_ + _e2460_);
                                                                let _e2473_ = (_e2472_ < 1536u);
                                                                if _e2473_ {
                                                                } else {
                                                                    phi_34453_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2472_] = _e2458_.member_2_;
                                                                let _e2476_ = (_e2459_ + 384u);
                                                                let _e2477_ = (_e2476_ + _e2460_);
                                                                let _e2478_ = (_e2477_ < 1536u);
                                                                if _e2478_ {
                                                                } else {
                                                                    phi_34453_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2477_] = _e2458_.member_3_;
                                                                let _e2481_ = (_e2459_ + 512u);
                                                                let _e2482_ = (_e2481_ + _e2460_);
                                                                let _e2483_ = (_e2482_ < 1536u);
                                                                if _e2483_ {
                                                                } else {
                                                                    phi_34453_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2482_] = _e2458_.member_4_;
                                                                let _e2486_ = (_e2459_ + 640u);
                                                                let _e2487_ = (_e2486_ + _e2460_);
                                                                let _e2488_ = (_e2487_ < 1536u);
                                                                if _e2488_ {
                                                                } else {
                                                                    phi_34453_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e2487_] = _e2458_.member_5_;
                                                                workgroupBarrier();
                                                                phi_3639_ = 1u;
                                                                phi_3642_ = _e2458_;
                                                                loop {
                                                                    let _e2491_ = phi_3639_;
                                                                    let _e2493_ = phi_3642_;
                                                                    let _e2494_ = (_e2491_ < 128u);
                                                                    if _e2494_ {
                                                                        if (_e170_ >= _e2491_) {
                                                                            let _e2497_ = ((_e170_ - _e2491_) & 127u);
                                                                            let _e2498_ = (_e2459_ + _e2497_);
                                                                            if (_e2498_ < 1536u) {
                                                                            } else {
                                                                                phi_34330_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2501_ = scan_scratch[_e2498_];
                                                                            let _e2502_ = (_e2466_ + _e2497_);
                                                                            if (_e2502_ < 1536u) {
                                                                            } else {
                                                                                phi_34330_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2505_ = scan_scratch[_e2502_];
                                                                            let _e2506_ = (_e2471_ + _e2497_);
                                                                            if (_e2506_ < 1536u) {
                                                                            } else {
                                                                                phi_34330_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2509_ = scan_scratch[_e2506_];
                                                                            let _e2510_ = (_e2476_ + _e2497_);
                                                                            if (_e2510_ < 1536u) {
                                                                            } else {
                                                                                phi_34330_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2513_ = scan_scratch[_e2510_];
                                                                            let _e2514_ = (_e2481_ + _e2497_);
                                                                            if (_e2514_ < 1536u) {
                                                                            } else {
                                                                                phi_34330_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2517_ = scan_scratch[_e2514_];
                                                                            let _e2518_ = (_e2486_ + _e2497_);
                                                                            if (_e2518_ < 1536u) {
                                                                            } else {
                                                                                phi_34330_ = true;
                                                                                break;
                                                                            }
                                                                            let _e2521_ = scan_scratch[_e2518_];
                                                                            phi_3744_ = type_32_(((_e2493_.member * _e2501_) + (_e2493_.member_1_ * _e2509_)), ((_e2493_.member * _e2505_) + (_e2493_.member_1_ * _e2513_)), ((_e2493_.member_2_ * _e2501_) + (_e2493_.member_3_ * _e2509_)), ((_e2493_.member_2_ * _e2505_) + (_e2493_.member_3_ * _e2513_)), (((_e2493_.member * _e2517_) + (_e2493_.member_1_ * _e2521_)) + _e2493_.member_4_), (((_e2493_.member_2_ * _e2517_) + (_e2493_.member_3_ * _e2521_)) + _e2493_.member_5_));
                                                                        } else {
                                                                            phi_3744_ = _e2493_;
                                                                        }
                                                                        let _e2550_ = phi_3744_;
                                                                        workgroupBarrier();
                                                                        if _e2463_ {
                                                                        } else {
                                                                            phi_34330_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2462_] = _e2550_.member;
                                                                        if _e2468_ {
                                                                        } else {
                                                                            phi_34330_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2467_] = _e2550_.member_1_;
                                                                        if _e2473_ {
                                                                        } else {
                                                                            phi_34330_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2472_] = _e2550_.member_2_;
                                                                        if _e2478_ {
                                                                        } else {
                                                                            phi_34330_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2477_] = _e2550_.member_3_;
                                                                        if _e2483_ {
                                                                        } else {
                                                                            phi_34330_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2482_] = _e2550_.member_4_;
                                                                        if _e2488_ {
                                                                        } else {
                                                                            phi_34330_ = true;
                                                                            break;
                                                                        }
                                                                        scan_scratch[_e2487_] = _e2550_.member_5_;
                                                                        workgroupBarrier();
                                                                        phi_3640_ = (_e2491_ << bitcast<u32>(1i));
                                                                        phi_3643_ = _e2550_;
                                                                    } else {
                                                                        phi_3640_ = u32();
                                                                        phi_3643_ = type_32_();
                                                                    }
                                                                    let _e2560_ = phi_3640_;
                                                                    let _e2562_ = phi_3643_;
                                                                    phi_3639_ = _e2560_;
                                                                    phi_3642_ = _e2562_;
                                                                    phi_34330_ = _e2376_;
                                                                    if !(_e2494_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e2565_ = phi_34330_;
                                                                phi_34453_ = _e2565_;
                                                                if _e2565_ {
                                                                    break;
                                                                }
                                                                if (_e170_ == 0u) {
                                                                    phi_3868_ = _e2374_;
                                                                } else {
                                                                    let _e2568_ = ((_e170_ - 1u) & 127u);
                                                                    let _e2569_ = (_e2459_ + _e2568_);
                                                                    if (_e2569_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2572_ = scan_scratch[_e2569_];
                                                                    let _e2573_ = (_e2466_ + _e2568_);
                                                                    if (_e2573_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2576_ = scan_scratch[_e2573_];
                                                                    let _e2577_ = (_e2471_ + _e2568_);
                                                                    if (_e2577_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2580_ = scan_scratch[_e2577_];
                                                                    let _e2581_ = (_e2476_ + _e2568_);
                                                                    if (_e2581_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2584_ = scan_scratch[_e2581_];
                                                                    let _e2585_ = (_e2481_ + _e2568_);
                                                                    if (_e2585_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2588_ = scan_scratch[_e2585_];
                                                                    let _e2589_ = (_e2486_ + _e2568_);
                                                                    if (_e2589_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2592_ = scan_scratch[_e2589_];
                                                                    phi_3868_ = array<f32, 2>((((_e2572_ * _e2369_) + (_e2576_ * _e2373_)) + _e2588_), (((_e2580_ * _e2369_) + (_e2584_ * _e2373_)) + _e2592_));
                                                                }
                                                                let _e2603_ = phi_3868_;
                                                                if _e1588_ {
                                                                    if (_e1587_ <= 128u) {
                                                                        phi_3880_ = (_e1587_ - 1u);
                                                                    } else {
                                                                        phi_3880_ = 127u;
                                                                    }
                                                                    let _e2609_ = phi_3880_;
                                                                    let _e2610_ = (_e2609_ & 127u);
                                                                    let _e2611_ = (_e2459_ + _e2610_);
                                                                    if (_e2611_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2614_ = scan_scratch[_e2611_];
                                                                    let _e2615_ = (_e2466_ + _e2610_);
                                                                    if (_e2615_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2618_ = scan_scratch[_e2615_];
                                                                    let _e2619_ = (_e2471_ + _e2610_);
                                                                    if (_e2619_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2622_ = scan_scratch[_e2619_];
                                                                    let _e2623_ = (_e2476_ + _e2610_);
                                                                    if (_e2623_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2626_ = scan_scratch[_e2623_];
                                                                    let _e2627_ = (_e2481_ + _e2610_);
                                                                    if (_e2627_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2630_ = scan_scratch[_e2627_];
                                                                    let _e2631_ = (_e2486_ + _e2610_);
                                                                    if (_e2631_ < 1536u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e2634_ = scan_scratch[_e2631_];
                                                                    phi_3944_ = array<f32, 2>((((_e2614_ * _e2369_) + (_e2618_ * _e2373_)) + _e2630_), (((_e2622_ * _e2369_) + (_e2626_ * _e2373_)) + _e2634_));
                                                                } else {
                                                                    phi_3944_ = _e2374_;
                                                                }
                                                                let _e2645_ = phi_3944_;
                                                                workgroupBarrier();
                                                                if _e1589_ {
                                                                    if ((bitcast<u32>(_e2603_[0]) & 2139095040u) != 2139095040u) {
                                                                        let _e2651_ = ((bitcast<u32>(_e2384_) & 2139095040u) != 2139095040u);
                                                                        if _e2651_ {
                                                                            let _e2652_ = abs(_e2603_[0]);
                                                                            let _e2653_ = abs(_e2384_);
                                                                            let _e2655_ = select(_e2653_, _e2652_, (_e2652_ > _e2653_));
                                                                            let _e2657_ = select(1f, _e2655_, (_e2655_ > 1f));
                                                                            if ((bitcast<u32>(_e2657_) & 2139095040u) != 2139095040u) {
                                                                                let _e2663_ = (_e2657_ <= 0f);
                                                                                if _e2663_ {
                                                                                    phi_4066_ = f32();
                                                                                } else {
                                                                                    if (_e2657_ < 0.00001f) {
                                                                                        let _e2695_ = (select(0.000000000000000000000001f, _e2657_, (_e2657_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                        if (_e2695_ != _e2695_) {
                                                                                            phi_24076_ = true;
                                                                                        } else {
                                                                                            phi_24076_ = (0.000001f >= _e2695_);
                                                                                        }
                                                                                        let _e2699_ = phi_24076_;
                                                                                        let _e2700_ = select(_e2695_, 0.000001f, _e2699_);
                                                                                        let _e2703_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2700_)));
                                                                                        let _e2707_ = (_e2703_ + (_e2703_ * (1f - (_e2700_ * _e2703_))));
                                                                                        let _e2711_ = (_e2707_ + (_e2707_ * (1f - (_e2700_ * _e2707_))));
                                                                                        let _e2717_ = bitcast<u32>(((_e2711_ + (_e2711_ * (1f - (_e2700_ * _e2711_)))) * 18446744000000000000f));
                                                                                        let _e2718_ = (_e2717_ & 2139095040u);
                                                                                        if (_e2718_ == 2139095040u) {
                                                                                            phi_4024_ = true;
                                                                                        } else {
                                                                                            phi_4024_ = (_e2718_ == 0u);
                                                                                        }
                                                                                        let _e2722_ = phi_4024_;
                                                                                        phi_4065_ = bitcast<f32>(select(_e2717_, 0u, _e2722_));
                                                                                    } else {
                                                                                        if (_e2657_ != _e2657_) {
                                                                                            phi_24016_ = true;
                                                                                        } else {
                                                                                            phi_24016_ = (0.000001f >= _e2657_);
                                                                                        }
                                                                                        let _e2668_ = phi_24016_;
                                                                                        let _e2669_ = select(_e2657_, 0.000001f, _e2668_);
                                                                                        let _e2672_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2669_)));
                                                                                        let _e2676_ = (_e2672_ + (_e2672_ * (1f - (_e2669_ * _e2672_))));
                                                                                        let _e2680_ = (_e2676_ + (_e2676_ * (1f - (_e2669_ * _e2676_))));
                                                                                        let _e2685_ = bitcast<u32>((_e2680_ + (_e2680_ * (1f - (_e2669_ * _e2680_)))));
                                                                                        let _e2686_ = (_e2685_ & 2139095040u);
                                                                                        if (_e2686_ == 2139095040u) {
                                                                                            phi_4059_ = true;
                                                                                        } else {
                                                                                            phi_4059_ = (_e2686_ == 0u);
                                                                                        }
                                                                                        let _e2690_ = phi_4059_;
                                                                                        phi_4065_ = bitcast<f32>(select(_e2685_, 0u, _e2690_));
                                                                                    }
                                                                                    let _e2726_ = phi_4065_;
                                                                                    phi_4066_ = _e2726_;
                                                                                }
                                                                                let _e2728_ = phi_4066_;
                                                                                phi_4067_ = _e2728_;
                                                                                phi_4068_ = _e2663_;
                                                                            } else {
                                                                                phi_4067_ = f32();
                                                                                phi_4068_ = true;
                                                                            }
                                                                            let _e2730_ = phi_4067_;
                                                                            let _e2732_ = phi_4068_;
                                                                            phi_4074_ = (abs((_e2603_[0] - _e2384_)) * select(_e2730_, 0f, _e2732_));
                                                                        } else {
                                                                            phi_4074_ = f32();
                                                                        }
                                                                        let _e2736_ = phi_4074_;
                                                                        phi_4076_ = _e2736_;
                                                                        phi_4077_ = select(true, false, _e2651_);
                                                                    } else {
                                                                        phi_4076_ = f32();
                                                                        phi_4077_ = true;
                                                                    }
                                                                    let _e2739_ = phi_4076_;
                                                                    let _e2741_ = phi_4077_;
                                                                    let _e2742_ = select(_e2739_, 1000000000f, _e2741_);
                                                                    if ((bitcast<u32>(_e2603_[1]) & 2139095040u) != 2139095040u) {
                                                                        let _e2748_ = ((bitcast<u32>(_e2382_) & 2139095040u) != 2139095040u);
                                                                        if _e2748_ {
                                                                            let _e2749_ = abs(_e2603_[1]);
                                                                            let _e2750_ = abs(_e2382_);
                                                                            let _e2752_ = select(_e2750_, _e2749_, (_e2749_ > _e2750_));
                                                                            let _e2754_ = select(1f, _e2752_, (_e2752_ > 1f));
                                                                            if ((bitcast<u32>(_e2754_) & 2139095040u) != 2139095040u) {
                                                                                let _e2760_ = (_e2754_ <= 0f);
                                                                                if _e2760_ {
                                                                                    phi_4200_ = f32();
                                                                                } else {
                                                                                    if (_e2754_ < 0.00001f) {
                                                                                        let _e2792_ = (select(0.000000000000000000000001f, _e2754_, (_e2754_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                        if (_e2792_ != _e2792_) {
                                                                                            phi_24196_ = true;
                                                                                        } else {
                                                                                            phi_24196_ = (0.000001f >= _e2792_);
                                                                                        }
                                                                                        let _e2796_ = phi_24196_;
                                                                                        let _e2797_ = select(_e2792_, 0.000001f, _e2796_);
                                                                                        let _e2800_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2797_)));
                                                                                        let _e2804_ = (_e2800_ + (_e2800_ * (1f - (_e2797_ * _e2800_))));
                                                                                        let _e2808_ = (_e2804_ + (_e2804_ * (1f - (_e2797_ * _e2804_))));
                                                                                        let _e2814_ = bitcast<u32>(((_e2808_ + (_e2808_ * (1f - (_e2797_ * _e2808_)))) * 18446744000000000000f));
                                                                                        let _e2815_ = (_e2814_ & 2139095040u);
                                                                                        if (_e2815_ == 2139095040u) {
                                                                                            phi_4158_ = true;
                                                                                        } else {
                                                                                            phi_4158_ = (_e2815_ == 0u);
                                                                                        }
                                                                                        let _e2819_ = phi_4158_;
                                                                                        phi_4199_ = bitcast<f32>(select(_e2814_, 0u, _e2819_));
                                                                                    } else {
                                                                                        if (_e2754_ != _e2754_) {
                                                                                            phi_24136_ = true;
                                                                                        } else {
                                                                                            phi_24136_ = (0.000001f >= _e2754_);
                                                                                        }
                                                                                        let _e2765_ = phi_24136_;
                                                                                        let _e2766_ = select(_e2754_, 0.000001f, _e2765_);
                                                                                        let _e2769_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2766_)));
                                                                                        let _e2773_ = (_e2769_ + (_e2769_ * (1f - (_e2766_ * _e2769_))));
                                                                                        let _e2777_ = (_e2773_ + (_e2773_ * (1f - (_e2766_ * _e2773_))));
                                                                                        let _e2782_ = bitcast<u32>((_e2777_ + (_e2777_ * (1f - (_e2766_ * _e2777_)))));
                                                                                        let _e2783_ = (_e2782_ & 2139095040u);
                                                                                        if (_e2783_ == 2139095040u) {
                                                                                            phi_4193_ = true;
                                                                                        } else {
                                                                                            phi_4193_ = (_e2783_ == 0u);
                                                                                        }
                                                                                        let _e2787_ = phi_4193_;
                                                                                        phi_4199_ = bitcast<f32>(select(_e2782_, 0u, _e2787_));
                                                                                    }
                                                                                    let _e2823_ = phi_4199_;
                                                                                    phi_4200_ = _e2823_;
                                                                                }
                                                                                let _e2825_ = phi_4200_;
                                                                                phi_4201_ = _e2825_;
                                                                                phi_4202_ = _e2760_;
                                                                            } else {
                                                                                phi_4201_ = f32();
                                                                                phi_4202_ = true;
                                                                            }
                                                                            let _e2827_ = phi_4201_;
                                                                            let _e2829_ = phi_4202_;
                                                                            phi_4208_ = (abs((_e2603_[1] - _e2382_)) * select(_e2827_, 0f, _e2829_));
                                                                        } else {
                                                                            phi_4208_ = f32();
                                                                        }
                                                                        let _e2833_ = phi_4208_;
                                                                        phi_4210_ = _e2833_;
                                                                        phi_4211_ = select(true, false, _e2748_);
                                                                    } else {
                                                                        phi_4210_ = f32();
                                                                        phi_4211_ = true;
                                                                    }
                                                                    let _e2836_ = phi_4210_;
                                                                    let _e2838_ = phi_4211_;
                                                                    let _e2839_ = select(_e2836_, 1000000000f, _e2838_);
                                                                    let _e2841_ = select(_e2742_, _e2839_, (_e2839_ > _e2742_));
                                                                    if (_e2403_.member_1_ > 0u) {
                                                                        let _e2844_ = ((_e170_ + 1u) == _e1587_);
                                                                        if _e2844_ {
                                                                            if ((bitcast<u32>(_e2645_[0]) & 2139095040u) != 2139095040u) {
                                                                                let _e2852_ = ((bitcast<u32>(_e2380_[0]) & 2139095040u) != 2139095040u);
                                                                                if _e2852_ {
                                                                                    let _e2853_ = abs(_e2645_[0]);
                                                                                    let _e2854_ = abs(_e2380_[0]);
                                                                                    let _e2856_ = select(_e2854_, _e2853_, (_e2853_ > _e2854_));
                                                                                    let _e2858_ = select(1f, _e2856_, (_e2856_ > 1f));
                                                                                    if ((bitcast<u32>(_e2858_) & 2139095040u) != 2139095040u) {
                                                                                        let _e2864_ = (_e2858_ <= 0f);
                                                                                        if _e2864_ {
                                                                                            phi_4350_ = f32();
                                                                                        } else {
                                                                                            if (_e2858_ < 0.00001f) {
                                                                                                let _e2896_ = (select(0.000000000000000000000001f, _e2858_, (_e2858_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                                if (_e2896_ != _e2896_) {
                                                                                                    phi_24316_ = true;
                                                                                                } else {
                                                                                                    phi_24316_ = (0.000001f >= _e2896_);
                                                                                                }
                                                                                                let _e2900_ = phi_24316_;
                                                                                                let _e2901_ = select(_e2896_, 0.000001f, _e2900_);
                                                                                                let _e2904_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2901_)));
                                                                                                let _e2908_ = (_e2904_ + (_e2904_ * (1f - (_e2901_ * _e2904_))));
                                                                                                let _e2912_ = (_e2908_ + (_e2908_ * (1f - (_e2901_ * _e2908_))));
                                                                                                let _e2918_ = bitcast<u32>(((_e2912_ + (_e2912_ * (1f - (_e2901_ * _e2912_)))) * 18446744000000000000f));
                                                                                                let _e2919_ = (_e2918_ & 2139095040u);
                                                                                                if (_e2919_ == 2139095040u) {
                                                                                                    phi_4308_ = true;
                                                                                                } else {
                                                                                                    phi_4308_ = (_e2919_ == 0u);
                                                                                                }
                                                                                                let _e2923_ = phi_4308_;
                                                                                                phi_4349_ = bitcast<f32>(select(_e2918_, 0u, _e2923_));
                                                                                            } else {
                                                                                                if (_e2858_ != _e2858_) {
                                                                                                    phi_24256_ = true;
                                                                                                } else {
                                                                                                    phi_24256_ = (0.000001f >= _e2858_);
                                                                                                }
                                                                                                let _e2869_ = phi_24256_;
                                                                                                let _e2870_ = select(_e2858_, 0.000001f, _e2869_);
                                                                                                let _e2873_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2870_)));
                                                                                                let _e2877_ = (_e2873_ + (_e2873_ * (1f - (_e2870_ * _e2873_))));
                                                                                                let _e2881_ = (_e2877_ + (_e2877_ * (1f - (_e2870_ * _e2877_))));
                                                                                                let _e2886_ = bitcast<u32>((_e2881_ + (_e2881_ * (1f - (_e2870_ * _e2881_)))));
                                                                                                let _e2887_ = (_e2886_ & 2139095040u);
                                                                                                if (_e2887_ == 2139095040u) {
                                                                                                    phi_4343_ = true;
                                                                                                } else {
                                                                                                    phi_4343_ = (_e2887_ == 0u);
                                                                                                }
                                                                                                let _e2891_ = phi_4343_;
                                                                                                phi_4349_ = bitcast<f32>(select(_e2886_, 0u, _e2891_));
                                                                                            }
                                                                                            let _e2927_ = phi_4349_;
                                                                                            phi_4350_ = _e2927_;
                                                                                        }
                                                                                        let _e2929_ = phi_4350_;
                                                                                        phi_4351_ = _e2929_;
                                                                                        phi_4352_ = _e2864_;
                                                                                    } else {
                                                                                        phi_4351_ = f32();
                                                                                        phi_4352_ = true;
                                                                                    }
                                                                                    let _e2931_ = phi_4351_;
                                                                                    let _e2933_ = phi_4352_;
                                                                                    phi_4358_ = (abs((_e2645_[0] - _e2380_[0])) * select(_e2931_, 0f, _e2933_));
                                                                                } else {
                                                                                    phi_4358_ = f32();
                                                                                }
                                                                                let _e2937_ = phi_4358_;
                                                                                phi_4360_ = _e2937_;
                                                                                phi_4361_ = select(true, false, _e2852_);
                                                                            } else {
                                                                                phi_4360_ = f32();
                                                                                phi_4361_ = true;
                                                                            }
                                                                            let _e2940_ = phi_4360_;
                                                                            let _e2942_ = phi_4361_;
                                                                            let _e2943_ = select(_e2940_, 1000000000f, _e2942_);
                                                                            if ((bitcast<u32>(_e2645_[1]) & 2139095040u) != 2139095040u) {
                                                                                let _e2951_ = ((bitcast<u32>(_e2380_[1]) & 2139095040u) != 2139095040u);
                                                                                if _e2951_ {
                                                                                    let _e2952_ = abs(_e2645_[1]);
                                                                                    let _e2953_ = abs(_e2380_[1]);
                                                                                    let _e2955_ = select(_e2953_, _e2952_, (_e2952_ > _e2953_));
                                                                                    let _e2957_ = select(1f, _e2955_, (_e2955_ > 1f));
                                                                                    if ((bitcast<u32>(_e2957_) & 2139095040u) != 2139095040u) {
                                                                                        let _e2963_ = (_e2957_ <= 0f);
                                                                                        if _e2963_ {
                                                                                            phi_4486_ = f32();
                                                                                        } else {
                                                                                            if (_e2957_ < 0.00001f) {
                                                                                                let _e2995_ = (select(0.000000000000000000000001f, _e2957_, (_e2957_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                                if (_e2995_ != _e2995_) {
                                                                                                    phi_24436_ = true;
                                                                                                } else {
                                                                                                    phi_24436_ = (0.000001f >= _e2995_);
                                                                                                }
                                                                                                let _e2999_ = phi_24436_;
                                                                                                let _e3000_ = select(_e2995_, 0.000001f, _e2999_);
                                                                                                let _e3003_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3000_)));
                                                                                                let _e3007_ = (_e3003_ + (_e3003_ * (1f - (_e3000_ * _e3003_))));
                                                                                                let _e3011_ = (_e3007_ + (_e3007_ * (1f - (_e3000_ * _e3007_))));
                                                                                                let _e3017_ = bitcast<u32>(((_e3011_ + (_e3011_ * (1f - (_e3000_ * _e3011_)))) * 18446744000000000000f));
                                                                                                let _e3018_ = (_e3017_ & 2139095040u);
                                                                                                if (_e3018_ == 2139095040u) {
                                                                                                    phi_4444_ = true;
                                                                                                } else {
                                                                                                    phi_4444_ = (_e3018_ == 0u);
                                                                                                }
                                                                                                let _e3022_ = phi_4444_;
                                                                                                phi_4485_ = bitcast<f32>(select(_e3017_, 0u, _e3022_));
                                                                                            } else {
                                                                                                if (_e2957_ != _e2957_) {
                                                                                                    phi_24376_ = true;
                                                                                                } else {
                                                                                                    phi_24376_ = (0.000001f >= _e2957_);
                                                                                                }
                                                                                                let _e2968_ = phi_24376_;
                                                                                                let _e2969_ = select(_e2957_, 0.000001f, _e2968_);
                                                                                                let _e2972_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2969_)));
                                                                                                let _e2976_ = (_e2972_ + (_e2972_ * (1f - (_e2969_ * _e2972_))));
                                                                                                let _e2980_ = (_e2976_ + (_e2976_ * (1f - (_e2969_ * _e2976_))));
                                                                                                let _e2985_ = bitcast<u32>((_e2980_ + (_e2980_ * (1f - (_e2969_ * _e2980_)))));
                                                                                                let _e2986_ = (_e2985_ & 2139095040u);
                                                                                                if (_e2986_ == 2139095040u) {
                                                                                                    phi_4479_ = true;
                                                                                                } else {
                                                                                                    phi_4479_ = (_e2986_ == 0u);
                                                                                                }
                                                                                                let _e2990_ = phi_4479_;
                                                                                                phi_4485_ = bitcast<f32>(select(_e2985_, 0u, _e2990_));
                                                                                            }
                                                                                            let _e3026_ = phi_4485_;
                                                                                            phi_4486_ = _e3026_;
                                                                                        }
                                                                                        let _e3028_ = phi_4486_;
                                                                                        phi_4487_ = _e3028_;
                                                                                        phi_4488_ = _e2963_;
                                                                                    } else {
                                                                                        phi_4487_ = f32();
                                                                                        phi_4488_ = true;
                                                                                    }
                                                                                    let _e3030_ = phi_4487_;
                                                                                    let _e3032_ = phi_4488_;
                                                                                    phi_4494_ = (abs((_e2645_[1] - _e2380_[1])) * select(_e3030_, 0f, _e3032_));
                                                                                } else {
                                                                                    phi_4494_ = f32();
                                                                                }
                                                                                let _e3036_ = phi_4494_;
                                                                                phi_4496_ = _e3036_;
                                                                                phi_4497_ = select(true, false, _e2951_);
                                                                            } else {
                                                                                phi_4496_ = f32();
                                                                                phi_4497_ = true;
                                                                            }
                                                                            let _e3039_ = phi_4496_;
                                                                            let _e3041_ = phi_4497_;
                                                                            let _e3042_ = select(_e3039_, 1000000000f, _e3041_);
                                                                            let _e3044_ = select(_e2943_, _e3042_, (_e3042_ > _e2943_));
                                                                            phi_4512_ = select(_e2841_, _e3044_, (_e3044_ > _e2841_));
                                                                        } else {
                                                                            phi_4512_ = f32();
                                                                        }
                                                                        let _e3048_ = phi_4512_;
                                                                        phi_4514_ = _e3048_;
                                                                        phi_4515_ = select(true, false, _e2844_);
                                                                    } else {
                                                                        phi_4514_ = f32();
                                                                        phi_4515_ = true;
                                                                    }
                                                                    let _e3051_ = phi_4514_;
                                                                    let _e3053_ = phi_4515_;
                                                                    phi_4520_ = select(_e3051_, _e2841_, _e3053_);
                                                                } else {
                                                                    phi_4520_ = 0f;
                                                                }
                                                                let _e3056_ = phi_4520_;
                                                                let _e3057_ = (_e169_ < 128u);
                                                                if _e3057_ {
                                                                    let _e3058_ = (32u + _e170_);
                                                                    if (_e3058_ < 256u) {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e3058_] = _e3056_;
                                                                }
                                                                workgroupBarrier();
                                                                if (_e169_ >= 128u) {
                                                                    let _e3062_ = (32u + _e170_);
                                                                    let _e3063_ = (_e3062_ < 256u);
                                                                    if _e3063_ {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    let _e3065_ = aux_scratch[_e3062_];
                                                                    if _e3063_ {
                                                                    } else {
                                                                        phi_34453_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e3062_] = select(_e3065_, _e3056_, (_e3056_ > _e3065_));
                                                                }
                                                                workgroupBarrier();
                                                                phi_4558_ = 64u;
                                                                loop {
                                                                    let _e3069_ = phi_4558_;
                                                                    let _e3070_ = (_e3069_ > 0u);
                                                                    if _e3070_ {
                                                                        if (_e170_ < _e3069_) {
                                                                            if _e3057_ {
                                                                                let _e3072_ = (32u + _e170_);
                                                                                let _e3073_ = (_e3072_ < 256u);
                                                                                if _e3073_ {
                                                                                } else {
                                                                                    phi_34362_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e3075_ = aux_scratch[_e3072_];
                                                                                let _e3076_ = (_e3072_ + _e3069_);
                                                                                if (_e3076_ < 256u) {
                                                                                } else {
                                                                                    phi_34362_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e3079_ = aux_scratch[_e3076_];
                                                                                if _e3073_ {
                                                                                } else {
                                                                                    phi_34362_ = true;
                                                                                    break;
                                                                                }
                                                                                aux_scratch[_e3072_] = select(_e3075_, _e3079_, (_e3079_ > _e3075_));
                                                                            }
                                                                        }
                                                                        workgroupBarrier();
                                                                        phi_4559_ = (_e3069_ >> bitcast<u32>(1i));
                                                                    } else {
                                                                        phi_4559_ = u32();
                                                                    }
                                                                    let _e3085_ = phi_4559_;
                                                                    phi_4558_ = _e3085_;
                                                                    phi_34362_ = _e2565_;
                                                                    if !(_e3070_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e3088_ = phi_34362_;
                                                                phi_34453_ = _e3088_;
                                                                if _e3088_ {
                                                                    break;
                                                                }
                                                                let _e3090_ = aux_scratch[32];
                                                                workgroupBarrier();
                                                                if (_e3090_ <= 0.0000005f) {
                                                                    if (_e2403_.member_1_ > 0u) {
                                                                        phi_4619_ = true;
                                                                    } else {
                                                                        phi_4619_ = (_e3090_ == 0f);
                                                                    }
                                                                    let _e3095_ = phi_4619_;
                                                                    phi_4624_ = select(_e2388_, true, _e3095_);
                                                                } else {
                                                                    phi_4624_ = _e2388_;
                                                                }
                                                                let _e3098_ = phi_4624_;
                                                                phi_34455_ = _e3088_;
                                                                phi_4625_ = _e2645_;
                                                                phi_4626_ = _e2603_[1];
                                                                phi_4627_ = _e2603_[0];
                                                                phi_4628_ = _e2645_;
                                                                phi_4629_ = _e3098_;
                                                            }
                                                            let _e3100_ = phi_34455_;
                                                            let _e3102_ = phi_4625_;
                                                            let _e3104_ = phi_4626_;
                                                            let _e3106_ = phi_4627_;
                                                            let _e3108_ = phi_4628_;
                                                            let _e3110_ = phi_4629_;
                                                            phi_34454_ = _e3100_;
                                                            phi_3498_ = _e3102_;
                                                            phi_3500_ = _e3104_;
                                                            phi_3502_ = _e3106_;
                                                            phi_3504_ = _e3108_;
                                                            phi_3506_ = _e3110_;
                                                        } else {
                                                            phi_34454_ = _e2376_;
                                                            phi_3498_ = array<f32, 2>();
                                                            phi_3500_ = f32();
                                                            phi_3502_ = f32();
                                                            phi_3504_ = array<f32, 2>();
                                                            phi_3506_ = bool();
                                                        }
                                                        let _e3112_ = phi_34454_;
                                                        let _e3114_ = phi_3498_;
                                                        let _e3116_ = phi_3500_;
                                                        let _e3118_ = phi_3502_;
                                                        let _e3120_ = phi_3504_;
                                                        let _e3122_ = phi_3506_;
                                                        phi_34344_ = _e3112_;
                                                        phi_3494_ = _e2401_;
                                                        phi_3497_ = _e3114_;
                                                        phi_3499_ = _e3116_;
                                                        phi_3501_ = _e3118_;
                                                        phi_3503_ = _e3120_;
                                                        phi_3505_ = _e3122_;
                                                        phi_34453_ = _e3112_;
                                                        if !(_e2407_) {
                                                            break;
                                                        }
                                                    }
                                                    let _e3125_ = phi_34453_;
                                                    phi_34543_ = _e3125_;
                                                    if _e3125_ {
                                                        break;
                                                    }
                                                    let _e3127_ = local_24_;
                                                    if _e3127_ {
                                                        if _e1589_ {
                                                            let _e3428_ = (_e2337_ * 0.5f);
                                                            if (_e3428_ != _e3428_) {
                                                                phi_24976_ = true;
                                                            } else {
                                                                phi_24976_ = (0.00001f >= _e3428_);
                                                            }
                                                            let _e3432_ = phi_24976_;
                                                            let _e3433_ = select(_e3428_, 0.00001f, _e3432_);
                                                            if (_e2295_ != _e2295_) {
                                                                phi_24991_ = true;
                                                            } else {
                                                                phi_24991_ = (0f >= _e2295_);
                                                            }
                                                            let _e3437_ = phi_24991_;
                                                            if (_e2344_ != _e2344_) {
                                                                phi_25006_ = true;
                                                            } else {
                                                                phi_25006_ = (0f >= _e2344_);
                                                            }
                                                            let _e3442_ = phi_25006_;
                                                            let _e3445_ = local_29_;
                                                            let _e3447_ = local_30_;
                                                            let _e3448_ = (_e3445_ * _e3447_);
                                                            if (_e3448_ != _e3448_) {
                                                                phi_25021_ = true;
                                                            } else {
                                                                phi_25021_ = (64f <= _e3448_);
                                                            }
                                                            let _e3452_ = phi_25021_;
                                                            let _e3455_ = (select(_e2295_, 0f, _e3437_) + (select(_e2344_, 0f, _e3442_) * select(_e3448_, 64f, _e3452_)));
                                                            let _e3458_ = (1f + (_e3433_ * (_e3433_ + _e3455_)));
                                                            let _e3461_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3458_)));
                                                            let _e3465_ = (_e3461_ + (_e3461_ * (1f - (_e3458_ * _e3461_))));
                                                            let _e3469_ = (_e3465_ + (_e3465_ * (1f - (_e3458_ * _e3465_))));
                                                            let _e3470_ = (_e3433_ * _e3469_);
                                                            let _e3474_ = local_31_;
                                                            let _e3475_ = (_e2363_ - _e3474_);
                                                            let _e3477_ = local_32_;
                                                            let _e3480_ = ((_e3469_ * _e3477_) + (_e3470_ * _e3475_));
                                                            let _e3482_ = local_33_;
                                                            let _e3485_ = local_34_;
                                                            let _e3488_ = ((_e3485_ + (_e3470_ * _e3482_)) + ((_e3433_ * _e3470_) * _e3475_));
                                                            let _e3503_ = select(0u, 2u, _e173_);
                                                            let _e3504_ = (_e2283_.member_1_ * 4u);
                                                            let _e3507_ = (_e170_ & 127u);
                                                            let _e3508_ = (((_e3504_ + _e3503_) * 128u) + _e3507_);
                                                            if (_e3508_ < 2048u) {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            lane_buf[_e3508_] = (((((_e3488_ * _e2302_) + (_e3480_ * _e2309_)) + (((_e2363_ - (_e3455_ * _e3480_)) - _e3488_) * _e2316_)) + ((_e2363_ - ((2f * _e3455_) * _e3480_)) * _e2323_)) + (_e2363_ * _e2330_));
                                                            let _e3514_ = (((_e3504_ + (_e3503_ + 1u)) * 128u) + _e3507_);
                                                            if (_e3514_ < 2048u) {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            lane_buf[_e3514_] = _e3480_;
                                                        }
                                                        if _e1190_ {
                                                            let _e3518_ = local_35_;
                                                            if (_e2283_.member_1_ < 256u) {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e2283_.member_1_] = _e3518_[0];
                                                            let _e3523_ = local_36_;
                                                            let _e3525_ = (4u + _e2283_.member_1_);
                                                            if (_e3525_ < 256u) {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e3525_] = _e3523_[1];
                                                        }
                                                        if (_e169_ == 128u) {
                                                            let _e3530_ = local_37_;
                                                            let _e3532_ = (8u + _e2283_.member_1_);
                                                            if (_e3532_ < 256u) {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e3532_] = _e3530_[0];
                                                            let _e3536_ = local_38_;
                                                            let _e3538_ = (12u + _e2283_.member_1_);
                                                            if (_e3538_ < 256u) {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e3538_] = _e3536_[1];
                                                        }
                                                        workgroupBarrier();
                                                        phi_34545_ = _e3125_;
                                                    } else {
                                                        let _e3128_ = select(0u, 128u, _e173_);
                                                        if _e1589_ {
                                                            let _e3133_ = (_e3128_ + _e170_);
                                                            if (_e3133_ < 1536u) {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            scan_scratch[_e3133_] = _e2363_;
                                                        } else {
                                                            if (_e170_ < 128u) {
                                                                let _e3130_ = (_e3128_ + _e170_);
                                                                if (_e3130_ < 1536u) {
                                                                } else {
                                                                    phi_34543_ = true;
                                                                    break;
                                                                }
                                                                scan_scratch[_e3130_] = 0f;
                                                            }
                                                        }
                                                        workgroupBarrier();
                                                        if _e1190_ {
                                                            let _e3136_ = (_e2283_.member_1_ < 256u);
                                                            if _e3136_ {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            let _e3138_ = aux_scratch[_e2283_.member_1_];
                                                            let _e3139_ = (4u + _e2283_.member_1_);
                                                            let _e3140_ = (_e3139_ < 256u);
                                                            if _e3140_ {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            let _e3142_ = aux_scratch[_e3139_];
                                                            phi_4992_ = type_15_(0u, _e1587_);
                                                            phi_4995_ = _e3142_;
                                                            phi_4997_ = _e3138_;
                                                            loop {
                                                                let _e3145_ = phi_4992_;
                                                                let _e3147_ = phi_4995_;
                                                                let _e3149_ = phi_4997_;
                                                                local_25_ = _e3149_;
                                                                local_26_ = _e3147_;
                                                                if (_e3145_.member < _e3145_.member_1_) {
                                                                    phi_4993_ = type_15_((_e3145_.member + 1u), _e3145_.member_1_);
                                                                    phi_5020_ = type_15_(1u, _e3145_.member);
                                                                } else {
                                                                    phi_4993_ = _e3145_;
                                                                    phi_5020_ = type_15_(0u, u32());
                                                                }
                                                                let _e3162_ = phi_4993_;
                                                                let _e3164_ = phi_5020_;
                                                                let _e3168_ = (bitcast<i32>(_e3164_.member) != 0i);
                                                                if _e3168_ {
                                                                    let _e3169_ = (_e3164_.member_1_ & 127u);
                                                                    if (_e3169_ < 1536u) {
                                                                    } else {
                                                                        phi_34459_ = true;
                                                                        break;
                                                                    }
                                                                    let _e3172_ = scan_scratch[_e3169_];
                                                                    let _e3173_ = (_e2337_ * 0.5f);
                                                                    if (_e3173_ != _e3173_) {
                                                                        phi_24496_ = true;
                                                                    } else {
                                                                        phi_24496_ = (0.00001f >= _e3173_);
                                                                    }
                                                                    let _e3177_ = phi_24496_;
                                                                    let _e3178_ = select(_e3173_, 0.00001f, _e3177_);
                                                                    if (_e2295_ != _e2295_) {
                                                                        phi_24556_ = true;
                                                                    } else {
                                                                        phi_24556_ = (0f >= _e2295_);
                                                                    }
                                                                    let _e3182_ = phi_24556_;
                                                                    if (_e2344_ != _e2344_) {
                                                                        phi_24616_ = true;
                                                                    } else {
                                                                        phi_24616_ = (0f >= _e2344_);
                                                                    }
                                                                    let _e3187_ = phi_24616_;
                                                                    let _e3189_ = (_e3149_ * _e3149_);
                                                                    if (_e3189_ != _e3189_) {
                                                                        phi_24676_ = true;
                                                                    } else {
                                                                        phi_24676_ = (64f <= _e3189_);
                                                                    }
                                                                    let _e3193_ = phi_24676_;
                                                                    let _e3196_ = (select(_e2295_, 0f, _e3182_) + (select(_e2344_, 0f, _e3187_) * select(_e3189_, 64f, _e3193_)));
                                                                    let _e3199_ = (1f + (_e3178_ * (_e3178_ + _e3196_)));
                                                                    let _e3202_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3199_)));
                                                                    let _e3206_ = (_e3202_ + (_e3202_ * (1f - (_e3199_ * _e3202_))));
                                                                    let _e3210_ = (_e3206_ + (_e3206_ * (1f - (_e3199_ * _e3206_))));
                                                                    let _e3211_ = (_e3178_ * _e3210_);
                                                                    let _e3214_ = (_e3172_ - _e3147_);
                                                                    let _e3217_ = ((_e3210_ * _e3149_) + (_e3211_ * _e3214_));
                                                                    let _e3221_ = ((_e3147_ + (_e3211_ * _e3149_)) + ((_e3178_ * _e3211_) * _e3214_));
                                                                    let _e3224_ = bitcast<u32>(((_e3217_ + _e3217_) - _e3149_));
                                                                    let _e3225_ = (_e3224_ & 2139095040u);
                                                                    if (_e3225_ == 2139095040u) {
                                                                        phi_5185_ = true;
                                                                    } else {
                                                                        phi_5185_ = (_e3225_ == 0u);
                                                                    }
                                                                    let _e3229_ = phi_5185_;
                                                                    let _e3234_ = bitcast<u32>(((_e3221_ + _e3221_) - _e3147_));
                                                                    let _e3235_ = (_e3234_ & 2139095040u);
                                                                    if (_e3235_ == 2139095040u) {
                                                                        phi_5239_ = true;
                                                                    } else {
                                                                        phi_5239_ = (_e3235_ == 0u);
                                                                    }
                                                                    let _e3239_ = phi_5239_;
                                                                    let _e3258_ = (_e3169_ & 127u);
                                                                    let _e3259_ = ((_e2283_.member_1_ * 512u) + _e3258_);
                                                                    if (_e3259_ < 2048u) {
                                                                    } else {
                                                                        phi_34459_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3259_] = (((((_e3221_ * _e2302_) + (_e3217_ * _e2309_)) + (((_e3172_ - (_e3196_ * _e3217_)) - _e3221_) * _e2316_)) + ((_e3172_ - ((_e3196_ + _e3196_) * _e3217_)) * _e2323_)) + (_e3172_ * _e2330_));
                                                                    let _e3264_ = ((((_e2283_.member_1_ * 4u) + 1u) * 128u) + _e3258_);
                                                                    if (_e3264_ < 2048u) {
                                                                    } else {
                                                                        phi_34459_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3264_] = _e3217_;
                                                                    phi_4996_ = bitcast<f32>(select(_e3234_, 0u, _e3239_));
                                                                    phi_4998_ = bitcast<f32>(select(_e3224_, 0u, _e3229_));
                                                                } else {
                                                                    phi_4996_ = f32();
                                                                    phi_4998_ = f32();
                                                                }
                                                                let _e3268_ = phi_4996_;
                                                                let _e3270_ = phi_4998_;
                                                                phi_4992_ = _e3162_;
                                                                phi_4995_ = _e3268_;
                                                                phi_4997_ = _e3270_;
                                                                phi_34459_ = _e3125_;
                                                                if !(_e3168_) {
                                                                    break;
                                                                }
                                                            }
                                                            let _e3273_ = phi_34459_;
                                                            phi_34543_ = _e3273_;
                                                            if _e3273_ {
                                                                break;
                                                            }
                                                            if _e3136_ {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            let _e3275_ = local_25_;
                                                            aux_scratch[_e2283_.member_1_] = _e3275_;
                                                            if _e3140_ {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            let _e3277_ = local_26_;
                                                            aux_scratch[_e3139_] = _e3277_;
                                                            phi_34506_ = _e3273_;
                                                        } else {
                                                            phi_34506_ = _e3125_;
                                                        }
                                                        let _e3279_ = phi_34506_;
                                                        if (_e169_ == 1u) {
                                                            let _e3281_ = (8u + _e2283_.member_1_);
                                                            let _e3282_ = (_e3281_ < 256u);
                                                            if _e3282_ {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            let _e3284_ = aux_scratch[_e3281_];
                                                            let _e3285_ = (12u + _e2283_.member_1_);
                                                            let _e3286_ = (_e3285_ < 256u);
                                                            if _e3286_ {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            let _e3288_ = aux_scratch[_e3285_];
                                                            phi_5375_ = type_15_(0u, _e1587_);
                                                            phi_5378_ = _e3288_;
                                                            phi_5380_ = _e3284_;
                                                            loop {
                                                                let _e3291_ = phi_5375_;
                                                                let _e3293_ = phi_5378_;
                                                                let _e3295_ = phi_5380_;
                                                                local_27_ = _e3295_;
                                                                local_28_ = _e3293_;
                                                                if (_e3291_.member < _e3291_.member_1_) {
                                                                    phi_5376_ = type_15_((_e3291_.member + 1u), _e3291_.member_1_);
                                                                    phi_5403_ = type_15_(1u, _e3291_.member);
                                                                } else {
                                                                    phi_5376_ = _e3291_;
                                                                    phi_5403_ = type_15_(0u, u32());
                                                                }
                                                                let _e3308_ = phi_5376_;
                                                                let _e3310_ = phi_5403_;
                                                                let _e3314_ = (bitcast<i32>(_e3310_.member) != 0i);
                                                                if _e3314_ {
                                                                    let _e3315_ = (_e3310_.member_1_ & 127u);
                                                                    let _e3316_ = (128u + _e3315_);
                                                                    if (_e3316_ < 1536u) {
                                                                    } else {
                                                                        phi_34502_ = true;
                                                                        break;
                                                                    }
                                                                    let _e3319_ = scan_scratch[_e3316_];
                                                                    let _e3320_ = (_e2337_ * 0.5f);
                                                                    if (_e3320_ != _e3320_) {
                                                                        phi_24736_ = true;
                                                                    } else {
                                                                        phi_24736_ = (0.00001f >= _e3320_);
                                                                    }
                                                                    let _e3324_ = phi_24736_;
                                                                    let _e3325_ = select(_e3320_, 0.00001f, _e3324_);
                                                                    if (_e2295_ != _e2295_) {
                                                                        phi_24796_ = true;
                                                                    } else {
                                                                        phi_24796_ = (0f >= _e2295_);
                                                                    }
                                                                    let _e3329_ = phi_24796_;
                                                                    if (_e2344_ != _e2344_) {
                                                                        phi_24856_ = true;
                                                                    } else {
                                                                        phi_24856_ = (0f >= _e2344_);
                                                                    }
                                                                    let _e3334_ = phi_24856_;
                                                                    let _e3336_ = (_e3295_ * _e3295_);
                                                                    if (_e3336_ != _e3336_) {
                                                                        phi_24916_ = true;
                                                                    } else {
                                                                        phi_24916_ = (64f <= _e3336_);
                                                                    }
                                                                    let _e3340_ = phi_24916_;
                                                                    let _e3343_ = (select(_e2295_, 0f, _e3329_) + (select(_e2344_, 0f, _e3334_) * select(_e3336_, 64f, _e3340_)));
                                                                    let _e3346_ = (1f + (_e3325_ * (_e3325_ + _e3343_)));
                                                                    let _e3349_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3346_)));
                                                                    let _e3353_ = (_e3349_ + (_e3349_ * (1f - (_e3346_ * _e3349_))));
                                                                    let _e3357_ = (_e3353_ + (_e3353_ * (1f - (_e3346_ * _e3353_))));
                                                                    let _e3358_ = (_e3325_ * _e3357_);
                                                                    let _e3361_ = (_e3319_ - _e3293_);
                                                                    let _e3364_ = ((_e3357_ * _e3295_) + (_e3358_ * _e3361_));
                                                                    let _e3368_ = ((_e3293_ + (_e3358_ * _e3295_)) + ((_e3325_ * _e3358_) * _e3361_));
                                                                    let _e3371_ = bitcast<u32>(((_e3364_ + _e3364_) - _e3295_));
                                                                    let _e3372_ = (_e3371_ & 2139095040u);
                                                                    if (_e3372_ == 2139095040u) {
                                                                        phi_5569_ = true;
                                                                    } else {
                                                                        phi_5569_ = (_e3372_ == 0u);
                                                                    }
                                                                    let _e3376_ = phi_5569_;
                                                                    let _e3381_ = bitcast<u32>(((_e3368_ + _e3368_) - _e3293_));
                                                                    let _e3382_ = (_e3381_ & 2139095040u);
                                                                    if (_e3382_ == 2139095040u) {
                                                                        phi_5623_ = true;
                                                                    } else {
                                                                        phi_5623_ = (_e3382_ == 0u);
                                                                    }
                                                                    let _e3386_ = phi_5623_;
                                                                    let _e3403_ = (_e2283_.member_1_ * 4u);
                                                                    let _e3406_ = (_e3315_ & 127u);
                                                                    let _e3407_ = (((_e3403_ + 2u) * 128u) + _e3406_);
                                                                    if (_e3407_ < 2048u) {
                                                                    } else {
                                                                        phi_34502_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3407_] = (((((_e3368_ * _e2302_) + (_e3364_ * _e2309_)) + (((_e3319_ - (_e3343_ * _e3364_)) - _e3368_) * _e2316_)) + ((_e3319_ - ((_e3343_ + _e3343_) * _e3364_)) * _e2323_)) + (_e3319_ * _e2330_));
                                                                    let _e3412_ = (((_e3403_ + 3u) * 128u) + _e3406_);
                                                                    if (_e3412_ < 2048u) {
                                                                    } else {
                                                                        phi_34502_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e3412_] = _e3364_;
                                                                    phi_5379_ = bitcast<f32>(select(_e3381_, 0u, _e3386_));
                                                                    phi_5381_ = bitcast<f32>(select(_e3371_, 0u, _e3376_));
                                                                } else {
                                                                    phi_5379_ = f32();
                                                                    phi_5381_ = f32();
                                                                }
                                                                let _e3416_ = phi_5379_;
                                                                let _e3418_ = phi_5381_;
                                                                phi_5375_ = _e3308_;
                                                                phi_5378_ = _e3416_;
                                                                phi_5380_ = _e3418_;
                                                                phi_34502_ = _e3279_;
                                                                if !(_e3314_) {
                                                                    break;
                                                                }
                                                            }
                                                            let _e3421_ = phi_34502_;
                                                            phi_34543_ = _e3421_;
                                                            if _e3421_ {
                                                                break;
                                                            }
                                                            if _e3282_ {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            let _e3423_ = local_27_;
                                                            aux_scratch[_e3281_] = _e3423_;
                                                            if _e3286_ {
                                                            } else {
                                                                phi_34543_ = true;
                                                                break;
                                                            }
                                                            let _e3425_ = local_28_;
                                                            aux_scratch[_e3285_] = _e3425_;
                                                            phi_34546_ = _e3421_;
                                                        } else {
                                                            phi_34546_ = _e3279_;
                                                        }
                                                        let _e3427_ = phi_34546_;
                                                        workgroupBarrier();
                                                        phi_34545_ = _e3427_;
                                                    }
                                                    let _e3542_ = phi_34545_;
                                                    phi_34544_ = _e3542_;
                                                } else {
                                                    phi_34544_ = _e2266_;
                                                }
                                                let _e3544_ = phi_34544_;
                                                phi_34356_ = _e3544_;
                                                phi_2943_ = _e2281_;
                                                phi_34543_ = _e3544_;
                                                if !(_e2287_) {
                                                    break;
                                                }
                                            }
                                            let _e3547_ = phi_34543_;
                                            phi_36576_ = _e3547_;
                                            if _e3547_ {
                                                break;
                                            }
                                            phi_34587_ = _e3547_;
                                        } else {
                                            if (_e169_ < 128u) {
                                                let _e1960_ = vec4(_e1589_);
                                                let _e1961_ = select(vec4<f32>(0f, 0f, 0f, 0f), _e1610_, _e1960_);
                                                let _e1962_ = select(vec4<f32>(0f, 0f, 0f, 0f), _e1608_, _e1960_);
                                                phi_5752_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e1964_ = phi_5752_;
                                                    if (_e1964_.member < _e1964_.member_1_) {
                                                        phi_5753_ = type_15_((_e1964_.member + 1u), _e1964_.member_1_);
                                                        phi_5776_ = type_15_(1u, _e1964_.member);
                                                    } else {
                                                        phi_5753_ = _e1964_;
                                                        phi_5776_ = type_15_(0u, u32());
                                                    }
                                                    let _e1977_ = phi_5753_;
                                                    let _e1979_ = phi_5776_;
                                                    let _e1983_ = (bitcast<i32>(_e1979_.member) != 0i);
                                                    if _e1983_ {
                                                        let _e1984_ = (_e1979_.member_1_ & 3u);
                                                        switch bitcast<i32>(_e1984_) {
                                                            case 0: {
                                                                phi_5794_ = _e1961_.x;
                                                                break;
                                                            }
                                                            case 1: {
                                                                phi_5794_ = _e1961_.y;
                                                                break;
                                                            }
                                                            case 2: {
                                                                phi_5794_ = _e1961_.z;
                                                                break;
                                                            }
                                                            default: {
                                                                phi_5794_ = _e1961_.w;
                                                                break;
                                                            }
                                                        }
                                                        let _e1991_ = phi_5794_;
                                                        let _e1992_ = bitcast<u32>(_e1991_);
                                                        let _e1993_ = (_e1992_ & 2139095040u);
                                                        if (_e1993_ == 2139095040u) {
                                                            phi_5805_ = true;
                                                        } else {
                                                            phi_5805_ = (_e1993_ == 0u);
                                                        }
                                                        let _e1997_ = phi_5805_;
                                                        let _e2002_ = ((_e1979_.member_1_ * 256u) + _e170_);
                                                        if (_e2002_ < 1536u) {
                                                        } else {
                                                            phi_34266_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e2002_] = bitcast<f32>(select(_e1992_, 0u, _e1997_));
                                                        switch bitcast<i32>(_e1984_) {
                                                            case 0: {
                                                                phi_5830_ = _e1962_.x;
                                                                break;
                                                            }
                                                            case 1: {
                                                                phi_5830_ = _e1962_.y;
                                                                break;
                                                            }
                                                            case 2: {
                                                                phi_5830_ = _e1962_.z;
                                                                break;
                                                            }
                                                            default: {
                                                                phi_5830_ = _e1962_.w;
                                                                break;
                                                            }
                                                        }
                                                        let _e2011_ = phi_5830_;
                                                        let _e2012_ = bitcast<u32>(_e2011_);
                                                        let _e2013_ = (_e2012_ & 2139095040u);
                                                        if (_e2013_ == 2139095040u) {
                                                            phi_5841_ = true;
                                                        } else {
                                                            phi_5841_ = (_e2013_ == 0u);
                                                        }
                                                        let _e2017_ = phi_5841_;
                                                        let _e2022_ = ((((_e1979_.member_1_ * 2u) + 1u) * 128u) + _e170_);
                                                        if (_e2022_ < 1536u) {
                                                        } else {
                                                            phi_34266_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e2022_] = bitcast<f32>(select(_e2012_, 0u, _e2017_));
                                                    }
                                                    phi_5752_ = _e1977_;
                                                    phi_34266_ = _e1558_;
                                                    if !(_e1983_) {
                                                        break;
                                                    }
                                                }
                                                let _e2027_ = phi_34266_;
                                                phi_36576_ = _e2027_;
                                                if _e2027_ {
                                                    break;
                                                }
                                                phi_34293_ = _e2027_;
                                            } else {
                                                phi_34293_ = _e1558_;
                                            }
                                            let _e2029_ = phi_34293_;
                                            workgroupBarrier();
                                            if (_e169_ < 8u) {
                                                let _e2032_ = (_e169_ >> bitcast<u32>(1i));
                                                let _e2033_ = (_e169_ & 1u);
                                                let _e2034_ = (_e2033_ != 0u);
                                                let _e2035_ = (_e2032_ & 3u);
                                                switch bitcast<i32>(_e2035_) {
                                                    case 0: {
                                                        phi_5969_ = _e333_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5969_ = _e333_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5969_ = _e333_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5969_ = _e333_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2042_ = phi_5969_;
                                                switch bitcast<i32>(_e2035_) {
                                                    case 0: {
                                                        phi_5986_ = _e347_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5986_ = _e347_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5986_ = _e347_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5986_ = _e347_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2049_ = phi_5986_;
                                                switch bitcast<i32>(_e2035_) {
                                                    case 0: {
                                                        phi_5997_ = _e354_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_5997_ = _e354_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_5997_ = _e354_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_5997_ = _e354_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2056_ = phi_5997_;
                                                switch bitcast<i32>(_e2035_) {
                                                    case 0: {
                                                        phi_6008_ = _e361_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6008_ = _e361_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6008_ = _e361_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6008_ = _e361_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2063_ = phi_6008_;
                                                switch bitcast<i32>(_e2035_) {
                                                    case 0: {
                                                        phi_6019_ = _e368_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6019_ = _e368_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6019_ = _e368_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6019_ = _e368_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2070_ = phi_6019_;
                                                switch bitcast<i32>(_e2035_) {
                                                    case 0: {
                                                        phi_6030_ = _e375_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6030_ = _e375_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6030_ = _e375_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6030_ = _e375_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2077_ = phi_6030_;
                                                switch bitcast<i32>(_e2035_) {
                                                    case 0: {
                                                        phi_6245_ = _e424_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6245_ = _e424_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6245_ = _e424_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6245_ = _e424_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2084_ = phi_6245_;
                                                switch bitcast<i32>(_e2035_) {
                                                    case 0: {
                                                        phi_6320_ = _e475_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6320_ = _e475_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6320_ = _e475_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6320_ = _e475_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2091_ = phi_6320_;
                                                if _e2034_ {
                                                    phi_6325_ = (8u + _e2032_);
                                                } else {
                                                    phi_6325_ = _e2032_;
                                                }
                                                let _e2094_ = phi_6325_;
                                                if _e2034_ {
                                                    phi_6331_ = (12u + _e2032_);
                                                } else {
                                                    phi_6331_ = (4u + _e2032_);
                                                }
                                                let _e2098_ = phi_6331_;
                                                let _e2099_ = (_e2094_ < 256u);
                                                if _e2099_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                let _e2101_ = aux_scratch[_e2094_];
                                                let _e2102_ = (_e2098_ < 256u);
                                                if _e2102_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                let _e2104_ = aux_scratch[_e2098_];
                                                let _e2108_ = select(0u, 2u, _e2034_);
                                                phi_6356_ = type_15_(0u, _e1587_);
                                                phi_6359_ = _e2104_;
                                                phi_6361_ = _e2101_;
                                                loop {
                                                    let _e2111_ = phi_6356_;
                                                    let _e2113_ = phi_6359_;
                                                    let _e2115_ = phi_6361_;
                                                    local_22_ = _e2115_;
                                                    local_23_ = _e2113_;
                                                    if (_e2111_.member < _e2111_.member_1_) {
                                                        phi_6357_ = type_15_((_e2111_.member + 1u), _e2111_.member_1_);
                                                        phi_6384_ = type_15_(1u, _e2111_.member);
                                                    } else {
                                                        phi_6357_ = _e2111_;
                                                        phi_6384_ = type_15_(0u, u32());
                                                    }
                                                    let _e2128_ = phi_6357_;
                                                    let _e2130_ = phi_6384_;
                                                    let _e2134_ = (bitcast<i32>(_e2130_.member) != 0i);
                                                    if _e2134_ {
                                                        let _e2135_ = (_e2130_.member_1_ & 127u);
                                                        let _e2136_ = ((((_e2032_ * 2u) + _e2033_) * 128u) + _e2135_);
                                                        if (_e2136_ < 1536u) {
                                                        } else {
                                                            phi_34278_ = true;
                                                            break;
                                                        }
                                                        let _e2139_ = scan_scratch[_e2136_];
                                                        let _e2140_ = (_e2084_ * 0.5f);
                                                        if (_e2140_ != _e2140_) {
                                                            phi_23716_ = true;
                                                        } else {
                                                            phi_23716_ = (0.00001f >= _e2140_);
                                                        }
                                                        let _e2144_ = phi_23716_;
                                                        let _e2145_ = select(_e2140_, 0.00001f, _e2144_);
                                                        if (_e2042_ != _e2042_) {
                                                            phi_23776_ = true;
                                                        } else {
                                                            phi_23776_ = (0f >= _e2042_);
                                                        }
                                                        let _e2149_ = phi_23776_;
                                                        if (_e2091_ != _e2091_) {
                                                            phi_23836_ = true;
                                                        } else {
                                                            phi_23836_ = (0f >= _e2091_);
                                                        }
                                                        let _e2154_ = phi_23836_;
                                                        let _e2156_ = (_e2115_ * _e2115_);
                                                        if (_e2156_ != _e2156_) {
                                                            phi_23896_ = true;
                                                        } else {
                                                            phi_23896_ = (64f <= _e2156_);
                                                        }
                                                        let _e2160_ = phi_23896_;
                                                        let _e2163_ = (select(_e2042_, 0f, _e2149_) + (select(_e2091_, 0f, _e2154_) * select(_e2156_, 64f, _e2160_)));
                                                        let _e2166_ = (1f + (_e2145_ * (_e2145_ + _e2163_)));
                                                        let _e2169_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2166_)));
                                                        let _e2173_ = (_e2169_ + (_e2169_ * (1f - (_e2166_ * _e2169_))));
                                                        let _e2177_ = (_e2173_ + (_e2173_ * (1f - (_e2166_ * _e2173_))));
                                                        let _e2178_ = (_e2145_ * _e2177_);
                                                        let _e2181_ = (_e2139_ - _e2113_);
                                                        let _e2184_ = ((_e2177_ * _e2115_) + (_e2178_ * _e2181_));
                                                        let _e2188_ = ((_e2113_ + (_e2178_ * _e2115_)) + ((_e2145_ * _e2178_) * _e2181_));
                                                        let _e2191_ = bitcast<u32>(((_e2184_ + _e2184_) - _e2115_));
                                                        let _e2192_ = (_e2191_ & 2139095040u);
                                                        if (_e2192_ == 2139095040u) {
                                                            phi_6550_ = true;
                                                        } else {
                                                            phi_6550_ = (_e2192_ == 0u);
                                                        }
                                                        let _e2196_ = phi_6550_;
                                                        let _e2201_ = bitcast<u32>(((_e2188_ + _e2188_) - _e2113_));
                                                        let _e2202_ = (_e2201_ & 2139095040u);
                                                        if (_e2202_ == 2139095040u) {
                                                            phi_6604_ = true;
                                                        } else {
                                                            phi_6604_ = (_e2202_ == 0u);
                                                        }
                                                        let _e2206_ = phi_6604_;
                                                        let _e2223_ = (_e2032_ * 4u);
                                                        let _e2226_ = (_e2135_ & 127u);
                                                        let _e2227_ = (((_e2223_ + _e2108_) * 128u) + _e2226_);
                                                        if (_e2227_ < 2048u) {
                                                        } else {
                                                            phi_34278_ = true;
                                                            break;
                                                        }
                                                        lane_buf[_e2227_] = (((((_e2188_ * _e2049_) + (_e2184_ * _e2056_)) + (((_e2139_ - (_e2163_ * _e2184_)) - _e2188_) * _e2063_)) + ((_e2139_ - ((_e2163_ + _e2163_) * _e2184_)) * _e2070_)) + (_e2139_ * _e2077_));
                                                        let _e2233_ = (((_e2223_ + (_e2108_ + 1u)) * 128u) + _e2226_);
                                                        if (_e2233_ < 2048u) {
                                                        } else {
                                                            phi_34278_ = true;
                                                            break;
                                                        }
                                                        lane_buf[_e2233_] = _e2184_;
                                                        phi_6360_ = bitcast<f32>(select(_e2201_, 0u, _e2206_));
                                                        phi_6362_ = bitcast<f32>(select(_e2191_, 0u, _e2196_));
                                                    } else {
                                                        phi_6360_ = f32();
                                                        phi_6362_ = f32();
                                                    }
                                                    let _e2237_ = phi_6360_;
                                                    let _e2239_ = phi_6362_;
                                                    phi_6356_ = _e2128_;
                                                    phi_6359_ = _e2237_;
                                                    phi_6361_ = _e2239_;
                                                    phi_34278_ = _e2029_;
                                                    if !(_e2134_) {
                                                        break;
                                                    }
                                                }
                                                let _e2242_ = phi_34278_;
                                                phi_36576_ = _e2242_;
                                                if _e2242_ {
                                                    break;
                                                }
                                                let _e2244_ = local_22_;
                                                let _e2245_ = bitcast<u32>(_e2244_);
                                                let _e2246_ = (_e2245_ & 2139095040u);
                                                if (_e2246_ == 2139095040u) {
                                                    phi_6716_ = true;
                                                } else {
                                                    phi_6716_ = (_e2246_ == 0u);
                                                }
                                                let _e2250_ = phi_6716_;
                                                if _e2099_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                aux_scratch[_e2094_] = bitcast<f32>(select(_e2245_, 0u, _e2250_));
                                                let _e2254_ = local_23_;
                                                let _e2255_ = bitcast<u32>(_e2254_);
                                                let _e2256_ = (_e2255_ & 2139095040u);
                                                if (_e2256_ == 2139095040u) {
                                                    phi_6738_ = true;
                                                } else {
                                                    phi_6738_ = (_e2256_ == 0u);
                                                }
                                                let _e2260_ = phi_6738_;
                                                if _e2102_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                aux_scratch[_e2098_] = bitcast<f32>(select(_e2255_, 0u, _e2260_));
                                                phi_34588_ = _e2242_;
                                            } else {
                                                phi_34588_ = _e2029_;
                                            }
                                            let _e2264_ = phi_34588_;
                                            workgroupBarrier();
                                            phi_34587_ = _e2264_;
                                        }
                                        let _e3549_ = phi_34587_;
                                        phi_34586_ = _e3549_;
                                    } else {
                                        phi_34154_ = _e1558_;
                                        phi_6752_ = type_15_(0u, 4u);
                                        loop {
                                            let _e1612_ = phi_34154_;
                                            let _e1614_ = phi_6752_;
                                            if (_e1614_.member < _e1614_.member_1_) {
                                                phi_6753_ = type_15_((_e1614_.member + 1u), _e1614_.member_1_);
                                                phi_6776_ = type_15_(1u, _e1614_.member);
                                            } else {
                                                phi_6753_ = _e1614_;
                                                phi_6776_ = type_15_(0u, u32());
                                            }
                                            let _e1627_ = phi_6753_;
                                            let _e1629_ = phi_6776_;
                                            let _e1633_ = (bitcast<i32>(_e1629_.member) != 0i);
                                            if _e1633_ {
                                                let _e1634_ = (_e1629_.member_1_ & 3u);
                                                switch bitcast<i32>(_e1634_) {
                                                    case 0: {
                                                        phi_6871_ = _e312_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6871_ = _e312_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6871_ = _e312_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6871_ = _e312_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1641_ = phi_6871_;
                                                switch bitcast<i32>(_e1634_) {
                                                    case 0: {
                                                        phi_6882_ = _e319_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6882_ = _e319_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6882_ = _e319_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6882_ = _e319_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1648_ = phi_6882_;
                                                switch bitcast<i32>(_e1634_) {
                                                    case 0: {
                                                        phi_6893_ = _e326_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6893_ = _e326_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6893_ = _e326_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6893_ = _e326_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1655_ = phi_6893_;
                                                switch bitcast<i32>(_e1634_) {
                                                    case 0: {
                                                        phi_6904_ = _e333_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6904_ = _e333_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6904_ = _e333_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6904_ = _e333_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1662_ = phi_6904_;
                                                switch bitcast<i32>(_e1634_) {
                                                    case 0: {
                                                        phi_6915_ = _e340_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6915_ = _e340_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6915_ = _e340_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6915_ = _e340_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1669_ = phi_6915_;
                                                switch bitcast<i32>(_e1634_) {
                                                    case 0: {
                                                        phi_6926_ = _e347_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6926_ = _e347_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6926_ = _e347_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6926_ = _e347_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1676_ = phi_6926_;
                                                switch bitcast<i32>(_e1634_) {
                                                    case 0: {
                                                        phi_6937_ = _e354_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6937_ = _e354_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6937_ = _e354_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6937_ = _e354_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1683_ = phi_6937_;
                                                switch bitcast<i32>(_e1634_) {
                                                    case 0: {
                                                        phi_6948_ = _e361_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6948_ = _e361_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6948_ = _e361_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6948_ = _e361_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1690_ = phi_6948_;
                                                switch bitcast<i32>(_e1634_) {
                                                    case 0: {
                                                        phi_6959_ = _e368_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6959_ = _e368_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6959_ = _e368_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6959_ = _e368_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1697_ = phi_6959_;
                                                switch bitcast<i32>(_e1634_) {
                                                    case 0: {
                                                        phi_6970_ = _e375_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6970_ = _e375_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6970_ = _e375_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6970_ = _e375_.w;
                                                        break;
                                                    }
                                                }
                                                let _e1704_ = phi_6970_;
                                                if _e1589_ {
                                                    let _e1706_ = select(_e1610_, _e1608_, vec4(_e173_));
                                                    switch bitcast<i32>(_e1634_) {
                                                        case 0: {
                                                            phi_7278_ = _e1706_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_7278_ = _e1706_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_7278_ = _e1706_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_7278_ = _e1706_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e1713_ = phi_7278_;
                                                    let _e1714_ = bitcast<u32>(_e1713_);
                                                    let _e1715_ = (_e1714_ & 2139095040u);
                                                    if (_e1715_ == 2139095040u) {
                                                        phi_7289_ = true;
                                                    } else {
                                                        phi_7289_ = (_e1715_ == 0u);
                                                    }
                                                    let _e1719_ = phi_7289_;
                                                    phi_7295_ = bitcast<f32>(select(_e1714_, 0u, _e1719_));
                                                } else {
                                                    phi_7295_ = 0f;
                                                }
                                                let _e1723_ = phi_7295_;
                                                if _e173_ {
                                                    phi_7300_ = (8u + _e1629_.member_1_);
                                                } else {
                                                    phi_7300_ = _e1629_.member_1_;
                                                }
                                                let _e1726_ = phi_7300_;
                                                if _e173_ {
                                                    phi_7306_ = (12u + _e1629_.member_1_);
                                                } else {
                                                    phi_7306_ = (4u + _e1629_.member_1_);
                                                }
                                                let _e1730_ = phi_7306_;
                                                if (_e1726_ < 256u) {
                                                } else {
                                                    phi_34249_ = true;
                                                    break;
                                                }
                                                let _e1733_ = aux_scratch[_e1726_];
                                                if (_e1730_ < 256u) {
                                                } else {
                                                    phi_34249_ = true;
                                                    break;
                                                }
                                                let _e1736_ = aux_scratch[_e1730_];
                                                let _e1738_ = select(0u, 256u, _e173_);
                                                let _e1739_ = (_e1648_ + _e1648_);
                                                let _e1740_ = (_e1655_ + _e1655_);
                                                let _e1742_ = ((_e1641_ + _e1641_) - 1f);
                                                let _e1743_ = -(_e1739_);
                                                let _e1744_ = (1f - _e1740_);
                                                if _e1589_ {
                                                    phi_7339_ = (_e1739_ * _e1723_);
                                                } else {
                                                    phi_7339_ = 0f;
                                                }
                                                let _e1747_ = phi_7339_;
                                                if _e1589_ {
                                                    phi_7344_ = (_e1740_ * _e1723_);
                                                } else {
                                                    phi_7344_ = 0f;
                                                }
                                                let _e1750_ = phi_7344_;
                                                let _e1751_ = (_e170_ == 0u);
                                                if _e1751_ {
                                                    if _e1588_ {
                                                        phi_7361_ = (_e1750_ + ((_e1739_ * _e1733_) + (_e1744_ * _e1736_)));
                                                        phi_7362_ = (_e1747_ + ((_e1742_ * _e1733_) + (_e1743_ * _e1736_)));
                                                    } else {
                                                        phi_7361_ = _e1750_;
                                                        phi_7362_ = _e1747_;
                                                    }
                                                    let _e1761_ = phi_7361_;
                                                    let _e1763_ = phi_7362_;
                                                    phi_7363_ = _e1761_;
                                                    phi_7364_ = _e1763_;
                                                } else {
                                                    phi_7363_ = _e1750_;
                                                    phi_7364_ = _e1747_;
                                                }
                                                let _e1765_ = phi_7363_;
                                                let _e1767_ = phi_7364_;
                                                let _e1768_ = (_e1738_ + _e170_);
                                                let _e1769_ = (_e1768_ < 1536u);
                                                if _e1769_ {
                                                } else {
                                                    phi_34249_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e1768_] = _e1767_;
                                                let _e1771_ = (_e1738_ + 128u);
                                                let _e1772_ = (_e1771_ + _e170_);
                                                let _e1773_ = (_e1772_ < 1536u);
                                                if _e1773_ {
                                                } else {
                                                    phi_34249_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e1772_] = _e1765_;
                                                workgroupBarrier();
                                                phi_7381_ = _e1744_;
                                                phi_7384_ = _e1739_;
                                                phi_7386_ = _e1743_;
                                                phi_7388_ = _e1742_;
                                                phi_7390_ = 1u;
                                                phi_7392_ = _e1765_;
                                                phi_7394_ = _e1767_;
                                                loop {
                                                    let _e1776_ = phi_7381_;
                                                    let _e1778_ = phi_7384_;
                                                    let _e1780_ = phi_7386_;
                                                    let _e1782_ = phi_7388_;
                                                    let _e1784_ = phi_7390_;
                                                    let _e1786_ = phi_7392_;
                                                    let _e1788_ = phi_7394_;
                                                    let _e1789_ = (_e1784_ < 128u);
                                                    if _e1789_ {
                                                        if (_e170_ >= _e1784_) {
                                                            let _e1792_ = ((_e170_ - _e1784_) & 127u);
                                                            let _e1793_ = (_e1738_ + _e1792_);
                                                            if (_e1793_ < 1536u) {
                                                            } else {
                                                                phi_34133_ = true;
                                                                break;
                                                            }
                                                            let _e1796_ = scan_scratch[_e1793_];
                                                            let _e1797_ = (_e1771_ + _e1792_);
                                                            if (_e1797_ < 1536u) {
                                                            } else {
                                                                phi_34133_ = true;
                                                                break;
                                                            }
                                                            let _e1800_ = scan_scratch[_e1797_];
                                                            phi_7433_ = (((_e1778_ * _e1796_) + (_e1776_ * _e1800_)) + _e1786_);
                                                            phi_7434_ = (((_e1782_ * _e1796_) + (_e1780_ * _e1800_)) + _e1788_);
                                                        } else {
                                                            phi_7433_ = _e1786_;
                                                            phi_7434_ = _e1788_;
                                                        }
                                                        let _e1810_ = phi_7433_;
                                                        let _e1812_ = phi_7434_;
                                                        workgroupBarrier();
                                                        if _e1769_ {
                                                        } else {
                                                            phi_34133_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1768_] = _e1812_;
                                                        if _e1773_ {
                                                        } else {
                                                            phi_34133_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e1772_] = _e1810_;
                                                        let _e1816_ = (_e1782_ + _e1776_);
                                                        workgroupBarrier();
                                                        phi_7382_ = ((_e1778_ * _e1780_) + (_e1776_ * _e1776_));
                                                        phi_7385_ = (_e1778_ * _e1816_);
                                                        phi_7387_ = (_e1780_ * _e1816_);
                                                        phi_7389_ = ((_e1782_ * _e1782_) + (_e1780_ * _e1778_));
                                                        phi_7391_ = (_e1784_ << bitcast<u32>(1i));
                                                        phi_7393_ = _e1810_;
                                                        phi_7395_ = _e1812_;
                                                    } else {
                                                        phi_7382_ = f32();
                                                        phi_7385_ = f32();
                                                        phi_7387_ = f32();
                                                        phi_7389_ = f32();
                                                        phi_7391_ = u32();
                                                        phi_7393_ = f32();
                                                        phi_7395_ = f32();
                                                    }
                                                    let _e1825_ = phi_7382_;
                                                    let _e1827_ = phi_7385_;
                                                    let _e1829_ = phi_7387_;
                                                    let _e1831_ = phi_7389_;
                                                    let _e1833_ = phi_7391_;
                                                    let _e1835_ = phi_7393_;
                                                    let _e1837_ = phi_7395_;
                                                    phi_7381_ = _e1825_;
                                                    phi_7384_ = _e1827_;
                                                    phi_7386_ = _e1829_;
                                                    phi_7388_ = _e1831_;
                                                    phi_7390_ = _e1833_;
                                                    phi_7392_ = _e1835_;
                                                    phi_7394_ = _e1837_;
                                                    phi_34133_ = _e1612_;
                                                    if !(_e1789_) {
                                                        break;
                                                    }
                                                }
                                                let _e1840_ = phi_34133_;
                                                phi_34249_ = _e1840_;
                                                if _e1840_ {
                                                    break;
                                                }
                                                if _e1751_ {
                                                    phi_7486_ = _e1736_;
                                                    phi_7487_ = _e1733_;
                                                } else {
                                                    let _e1842_ = ((_e170_ - 1u) & 127u);
                                                    let _e1843_ = (_e1738_ + _e1842_);
                                                    if (_e1843_ < 1536u) {
                                                    } else {
                                                        phi_34249_ = true;
                                                        break;
                                                    }
                                                    let _e1846_ = scan_scratch[_e1843_];
                                                    let _e1847_ = (_e1771_ + _e1842_);
                                                    if (_e1847_ < 1536u) {
                                                    } else {
                                                        phi_34249_ = true;
                                                        break;
                                                    }
                                                    let _e1850_ = scan_scratch[_e1847_];
                                                    phi_7486_ = _e1850_;
                                                    phi_7487_ = _e1846_;
                                                }
                                                let _e1852_ = phi_7486_;
                                                let _e1854_ = phi_7487_;
                                                if _e1588_ {
                                                    if (_e1587_ <= 128u) {
                                                        phi_7497_ = (_e1587_ - 1u);
                                                    } else {
                                                        phi_7497_ = 127u;
                                                    }
                                                    let _e1858_ = phi_7497_;
                                                    let _e1859_ = (_e1738_ + _e1858_);
                                                    if (_e1859_ < 1536u) {
                                                    } else {
                                                        phi_34249_ = true;
                                                        break;
                                                    }
                                                    let _e1862_ = scan_scratch[_e1859_];
                                                    let _e1863_ = (_e1771_ + _e1858_);
                                                    if (_e1863_ < 1536u) {
                                                    } else {
                                                        phi_34249_ = true;
                                                        break;
                                                    }
                                                    let _e1866_ = scan_scratch[_e1863_];
                                                    phi_7516_ = array<f32, 2>(_e1862_, _e1866_);
                                                } else {
                                                    phi_7516_ = array<f32, 2>(_e1733_, _e1736_);
                                                }
                                                let _e1869_ = phi_7516_;
                                                workgroupBarrier();
                                                if _e1589_ {
                                                    let _e1870_ = (_e1723_ - _e1852_);
                                                    let _e1873_ = ((_e1641_ * _e1854_) + (_e1648_ * _e1870_));
                                                    let _e1877_ = ((_e1852_ + (_e1648_ * _e1854_)) + (_e1655_ * _e1870_));
                                                    let _e1892_ = select(0u, 2u, _e173_);
                                                    let _e1893_ = (_e1629_.member_1_ * 4u);
                                                    let _e1896_ = (_e170_ & 127u);
                                                    let _e1897_ = (((_e1893_ + _e1892_) * 128u) + _e1896_);
                                                    if (_e1897_ < 2048u) {
                                                    } else {
                                                        phi_34249_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e1897_] = (((((_e1877_ * _e1676_) + (_e1873_ * _e1683_)) + (((_e1723_ - (_e1662_ * _e1873_)) - _e1877_) * _e1690_)) + ((_e1723_ - (_e1669_ * _e1873_)) * _e1697_)) + (_e1723_ * _e1704_));
                                                    let _e1903_ = (((_e1893_ + (_e1892_ + 1u)) * 128u) + _e1896_);
                                                    if (_e1903_ < 2048u) {
                                                    } else {
                                                        phi_34249_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e1903_] = _e1873_;
                                                }
                                                if _e1190_ {
                                                    let _e1907_ = bitcast<u32>(_e1869_[0]);
                                                    let _e1908_ = (_e1907_ & 2139095040u);
                                                    if (_e1908_ == 2139095040u) {
                                                        phi_7771_ = true;
                                                    } else {
                                                        phi_7771_ = (_e1908_ == 0u);
                                                    }
                                                    let _e1912_ = phi_7771_;
                                                    if (_e1629_.member_1_ < 256u) {
                                                    } else {
                                                        phi_34249_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1629_.member_1_] = bitcast<f32>(select(_e1907_, 0u, _e1912_));
                                                    let _e1918_ = bitcast<u32>(_e1869_[1]);
                                                    let _e1919_ = (_e1918_ & 2139095040u);
                                                    if (_e1919_ == 2139095040u) {
                                                        phi_7794_ = true;
                                                    } else {
                                                        phi_7794_ = (_e1919_ == 0u);
                                                    }
                                                    let _e1923_ = phi_7794_;
                                                    let _e1926_ = (4u + _e1629_.member_1_);
                                                    if (_e1926_ < 256u) {
                                                    } else {
                                                        phi_34249_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1926_] = bitcast<f32>(select(_e1918_, 0u, _e1923_));
                                                }
                                                if (_e169_ == 128u) {
                                                    let _e1931_ = bitcast<u32>(_e1869_[0]);
                                                    let _e1932_ = (_e1931_ & 2139095040u);
                                                    if (_e1932_ == 2139095040u) {
                                                        phi_7822_ = true;
                                                    } else {
                                                        phi_7822_ = (_e1932_ == 0u);
                                                    }
                                                    let _e1936_ = phi_7822_;
                                                    let _e1939_ = (8u + _e1629_.member_1_);
                                                    if (_e1939_ < 256u) {
                                                    } else {
                                                        phi_34249_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1939_] = bitcast<f32>(select(_e1931_, 0u, _e1936_));
                                                    let _e1943_ = bitcast<u32>(_e1869_[1]);
                                                    let _e1944_ = (_e1943_ & 2139095040u);
                                                    if (_e1944_ == 2139095040u) {
                                                        phi_7846_ = true;
                                                    } else {
                                                        phi_7846_ = (_e1944_ == 0u);
                                                    }
                                                    let _e1948_ = phi_7846_;
                                                    let _e1951_ = (12u + _e1629_.member_1_);
                                                    if (_e1951_ < 256u) {
                                                    } else {
                                                        phi_34249_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e1951_] = bitcast<f32>(select(_e1943_, 0u, _e1948_));
                                                }
                                                workgroupBarrier();
                                                phi_34250_ = _e1840_;
                                            } else {
                                                phi_34250_ = _e1612_;
                                            }
                                            let _e1955_ = phi_34250_;
                                            phi_34154_ = _e1955_;
                                            phi_6752_ = _e1627_;
                                            phi_34249_ = _e1955_;
                                            if !(_e1633_) {
                                                break;
                                            }
                                        }
                                        let _e1958_ = phi_34249_;
                                        phi_36576_ = _e1958_;
                                        if _e1958_ {
                                            break;
                                        }
                                        phi_34586_ = _e1958_;
                                    }
                                    let _e3551_ = phi_34586_;
                                    phi_34585_ = _e3551_;
                                    phi_7861_ = type_15_(0u, 4u);
                                    loop {
                                        let _e3553_ = phi_34585_;
                                        let _e3555_ = phi_7861_;
                                        if (_e3555_.member < _e3555_.member_1_) {
                                            phi_7862_ = type_15_((_e3555_.member + 1u), _e3555_.member_1_);
                                            phi_7885_ = type_15_(1u, _e3555_.member);
                                        } else {
                                            phi_7862_ = _e3555_;
                                            phi_7885_ = type_15_(0u, u32());
                                        }
                                        let _e3568_ = phi_7862_;
                                        let _e3570_ = phi_7885_;
                                        let _e3574_ = (bitcast<i32>(_e3570_.member) != 0i);
                                        if _e3574_ {
                                            let _e3575_ = (_e3570_.member_1_ & 3u);
                                            switch bitcast<i32>(_e3575_) {
                                                case 0: {
                                                    phi_7903_ = _e382_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_7903_ = _e382_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_7903_ = _e382_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_7903_ = _e382_.w;
                                                    break;
                                                }
                                            }
                                            let _e3582_ = phi_7903_;
                                            switch bitcast<i32>(_e3575_) {
                                                case 0: {
                                                    phi_7914_ = _e389_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_7914_ = _e389_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_7914_ = _e389_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_7914_ = _e389_.w;
                                                    break;
                                                }
                                            }
                                            let _e3589_ = phi_7914_;
                                            switch bitcast<i32>(_e3575_) {
                                                case 0: {
                                                    phi_7925_ = _e396_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_7925_ = _e396_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_7925_ = _e396_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_7925_ = _e396_.w;
                                                    break;
                                                }
                                            }
                                            let _e3596_ = phi_7925_;
                                            switch bitcast<i32>(_e3575_) {
                                                case 0: {
                                                    phi_7936_ = _e403_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_7936_ = _e403_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_7936_ = _e403_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_7936_ = _e403_.w;
                                                    break;
                                                }
                                            }
                                            let _e3603_ = phi_7936_;
                                            switch bitcast<i32>(_e3575_) {
                                                case 0: {
                                                    phi_7947_ = _e410_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_7947_ = _e410_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_7947_ = _e410_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_7947_ = _e410_.w;
                                                    break;
                                                }
                                            }
                                            let _e3610_ = phi_7947_;
                                            let _e3611_ = bitcast<u32>(_e3582_);
                                            let _e3612_ = (_e3611_ & 2139095040u);
                                            if (_e3612_ != 2139095040u) {
                                                if (_e3582_ <= 0f) {
                                                    phi_8061_ = 0f;
                                                } else {
                                                    if (_e3582_ >= 0.999995f) {
                                                        phi_8060_ = 0.999995f;
                                                    } else {
                                                        if (_e3612_ == 2139095040u) {
                                                            phi_8054_ = true;
                                                        } else {
                                                            phi_8054_ = (_e3612_ == 0u);
                                                        }
                                                        let _e3621_ = phi_8054_;
                                                        phi_8060_ = bitcast<f32>(select(_e3611_, 0u, _e3621_));
                                                    }
                                                    let _e3625_ = phi_8060_;
                                                    phi_8061_ = _e3625_;
                                                }
                                                let _e3627_ = phi_8061_;
                                                phi_8068_ = _e3627_;
                                            } else {
                                                phi_8068_ = select(0f, 0.999995f, (_e3611_ == 2139095040u));
                                            }
                                            let _e3629_ = phi_8068_;
                                            let _e3630_ = bitcast<u32>(_e3589_);
                                            let _e3631_ = (_e3630_ & 2139095040u);
                                            if (_e3631_ != 2139095040u) {
                                                if (_e3589_ <= 0f) {
                                                    phi_8100_ = 0f;
                                                } else {
                                                    if (_e3589_ >= 1f) {
                                                        phi_8099_ = 1f;
                                                    } else {
                                                        if (_e3631_ == 2139095040u) {
                                                            phi_8093_ = true;
                                                        } else {
                                                            phi_8093_ = (_e3631_ == 0u);
                                                        }
                                                        let _e3640_ = phi_8093_;
                                                        phi_8099_ = bitcast<f32>(select(_e3630_, 0u, _e3640_));
                                                    }
                                                    let _e3644_ = phi_8099_;
                                                    phi_8100_ = _e3644_;
                                                }
                                                let _e3646_ = phi_8100_;
                                                phi_8107_ = _e3646_;
                                            } else {
                                                phi_8107_ = select(0f, 1f, (_e3630_ == 2139095040u));
                                            }
                                            let _e3648_ = phi_8107_;
                                            let _e3649_ = bitcast<u32>(_e3596_);
                                            let _e3650_ = (_e3649_ & 2139095040u);
                                            if (_e3650_ != 2139095040u) {
                                                if (_e3596_ <= 0f) {
                                                    phi_8139_ = 0f;
                                                } else {
                                                    if (_e3596_ >= 1f) {
                                                        phi_8138_ = 1f;
                                                    } else {
                                                        if (_e3650_ == 2139095040u) {
                                                            phi_8132_ = true;
                                                        } else {
                                                            phi_8132_ = (_e3650_ == 0u);
                                                        }
                                                        let _e3659_ = phi_8132_;
                                                        phi_8138_ = bitcast<f32>(select(_e3649_, 0u, _e3659_));
                                                    }
                                                    let _e3663_ = phi_8138_;
                                                    phi_8139_ = _e3663_;
                                                }
                                                let _e3665_ = phi_8139_;
                                                phi_8146_ = _e3665_;
                                            } else {
                                                phi_8146_ = select(0f, 1f, (_e3649_ == 2139095040u));
                                            }
                                            let _e3667_ = phi_8146_;
                                            let _e3668_ = bitcast<u32>(_e3603_);
                                            let _e3669_ = (_e3668_ & 2139095040u);
                                            if (_e3669_ != 2139095040u) {
                                                if (_e3603_ <= 0f) {
                                                    phi_8178_ = 0f;
                                                } else {
                                                    if (_e3603_ >= 64f) {
                                                        phi_8177_ = 64f;
                                                    } else {
                                                        if (_e3669_ == 2139095040u) {
                                                            phi_8171_ = true;
                                                        } else {
                                                            phi_8171_ = (_e3669_ == 0u);
                                                        }
                                                        let _e3678_ = phi_8171_;
                                                        phi_8177_ = bitcast<f32>(select(_e3668_, 0u, _e3678_));
                                                    }
                                                    let _e3682_ = phi_8177_;
                                                    phi_8178_ = _e3682_;
                                                }
                                                let _e3684_ = phi_8178_;
                                                phi_8185_ = _e3684_;
                                            } else {
                                                phi_8185_ = select(0f, 64f, (_e3668_ == 2139095040u));
                                            }
                                            let _e3686_ = phi_8185_;
                                            let _e3687_ = bitcast<u32>(_e3610_);
                                            let _e3688_ = (_e3687_ & 2139095040u);
                                            if (_e3688_ != 2139095040u) {
                                                if (_e3610_ <= 0f) {
                                                    phi_8217_ = 0f;
                                                } else {
                                                    if (_e3610_ >= 1f) {
                                                        phi_8216_ = 1f;
                                                    } else {
                                                        if (_e3688_ == 2139095040u) {
                                                            phi_8210_ = true;
                                                        } else {
                                                            phi_8210_ = (_e3688_ == 0u);
                                                        }
                                                        let _e3697_ = phi_8210_;
                                                        phi_8216_ = bitcast<f32>(select(_e3687_, 0u, _e3697_));
                                                    }
                                                    let _e3701_ = phi_8216_;
                                                    phi_8217_ = _e3701_;
                                                }
                                                let _e3703_ = phi_8217_;
                                                phi_8224_ = _e3703_;
                                            } else {
                                                phi_8224_ = select(0f, 1f, (_e3687_ == 2139095040u));
                                            }
                                            let _e3705_ = phi_8224_;
                                            switch bitcast<i32>(_e3575_) {
                                                case 0: {
                                                    phi_8319_ = _e424_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8319_ = _e424_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8319_ = _e424_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8319_ = _e424_.w;
                                                    break;
                                                }
                                            }
                                            let _e3712_ = phi_8319_;
                                            switch bitcast<i32>(_e3575_) {
                                                case 0: {
                                                    phi_8330_ = _e431_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8330_ = _e431_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8330_ = _e431_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8330_ = _e431_.w;
                                                    break;
                                                }
                                            }
                                            let _e3719_ = phi_8330_;
                                            if (_e3667_ == 0f) {
                                                phi_8400_ = true;
                                            } else {
                                                phi_8400_ = (_e3705_ >= 1f);
                                            }
                                            let _e3723_ = phi_8400_;
                                            if _e3723_ {
                                                let _e3724_ = select(0u, 2u, _e173_);
                                                let _e3725_ = (_e3724_ + 1u);
                                                if _e1589_ {
                                                    let _e3726_ = (_e3570_.member_1_ * 4u);
                                                    let _e3729_ = (_e170_ & 127u);
                                                    let _e3730_ = (((_e3726_ + _e3724_) * 128u) + _e3729_);
                                                    if (_e3730_ < 2048u) {
                                                    } else {
                                                        phi_34710_ = true;
                                                        break;
                                                    }
                                                    let _e3733_ = lane_buf[_e3730_];
                                                    let _e3736_ = (((_e3726_ + _e3725_) * 128u) + _e3729_);
                                                    if (_e3736_ < 2048u) {
                                                    } else {
                                                        phi_34710_ = true;
                                                        break;
                                                    }
                                                    let _e3739_ = lane_buf[_e3736_];
                                                    let _e3741_ = bitcast<u32>(_e3739_);
                                                    let _e3742_ = (_e3741_ & 2139095040u);
                                                    if (_e3742_ == 2139095040u) {
                                                        phi_8447_ = true;
                                                    } else {
                                                        phi_8447_ = (_e3742_ == 0u);
                                                    }
                                                    let _e3746_ = phi_8447_;
                                                    let _e3750_ = bitcast<u32>((_e3712_ * bitcast<f32>(select(_e3741_, 0u, _e3746_))));
                                                    let _e3751_ = (_e3750_ & 2139095040u);
                                                    if (_e3751_ == 2139095040u) {
                                                        phi_8464_ = true;
                                                    } else {
                                                        phi_8464_ = (_e3751_ == 0u);
                                                    }
                                                    let _e3755_ = phi_8464_;
                                                    if (_e3686_ <= 0f) {
                                                        phi_8476_ = 0f;
                                                    } else {
                                                        phi_8476_ = (_e3686_ * abs(bitcast<f32>(select(_e3750_, 0u, _e3755_))));
                                                    }
                                                    let _e3762_ = phi_8476_;
                                                    let _e3764_ = ((1f - _e3648_) + _e3762_);
                                                    if (_e3764_ <= 0f) {
                                                        phi_8488_ = 0f;
                                                    } else {
                                                        phi_8488_ = select(_e3764_, 1f, (_e3764_ >= 1f));
                                                    }
                                                    let _e3769_ = phi_8488_;
                                                    let _e3770_ = bitcast<u32>((_e3719_ * _e3733_));
                                                    let _e3771_ = (_e3770_ & 2139095040u);
                                                    if (_e3771_ == 2139095040u) {
                                                        phi_8499_ = true;
                                                    } else {
                                                        phi_8499_ = (_e3771_ == 0u);
                                                    }
                                                    let _e3775_ = phi_8499_;
                                                    let _e3777_ = bitcast<f32>(select(_e3770_, 0u, _e3775_));
                                                    let _e3780_ = bitcast<f32>((bitcast<u32>(_e3777_) & 2147483647u));
                                                    if (_e3780_ != _e3780_) {
                                                        phi_25036_ = true;
                                                    } else {
                                                        phi_25036_ = (1000000000000000f <= _e3780_);
                                                    }
                                                    let _e3784_ = phi_25036_;
                                                    let _e3785_ = select(_e3780_, 1000000000000000f, _e3784_);
                                                    let _e3786_ = (_e3785_ * _e3785_);
                                                    let _e3787_ = (_e3786_ + 0.0000010000001f);
                                                    if (_e3787_ != _e3787_) {
                                                        phi_25096_ = true;
                                                    } else {
                                                        phi_25096_ = (0.000000000001f >= _e3787_);
                                                    }
                                                    let _e3791_ = phi_25096_;
                                                    let _e3792_ = select(_e3787_, 0.000000000001f, _e3791_);
                                                    let _e3793_ = (_e3792_ * 0.5f);
                                                    let _e3798_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e3792_) >> bitcast<u32>(1i))));
                                                    let _e3802_ = (_e3798_ * (1.5f - (_e3793_ * (_e3798_ * _e3798_))));
                                                    let _e3806_ = (_e3802_ * (1.5f - (_e3793_ * (_e3802_ * _e3802_))));
                                                    let _e3811_ = (_e3787_ * (_e3806_ * (1.5f - (_e3793_ * (_e3806_ * _e3806_)))));
                                                    let _e3812_ = (_e3811_ + 0.001f);
                                                    if (_e3812_ != _e3812_) {
                                                        phi_25156_ = true;
                                                    } else {
                                                        phi_25156_ = (0.000001f >= _e3812_);
                                                    }
                                                    let _e3816_ = phi_25156_;
                                                    let _e3817_ = select(_e3812_, 0.000001f, _e3816_);
                                                    let _e3820_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3817_)));
                                                    let _e3824_ = (_e3820_ + (_e3820_ * (1f - (_e3817_ * _e3820_))));
                                                    let _e3828_ = (_e3824_ + (_e3824_ * (1f - (_e3817_ * _e3824_))));
                                                    let _e3832_ = (_e3828_ + (_e3828_ * (1f - (_e3817_ * _e3828_))));
                                                    let _e3833_ = (_e3811_ + _e3785_);
                                                    if (_e3833_ != _e3833_) {
                                                        phi_25216_ = true;
                                                    } else {
                                                        phi_25216_ = (0.000001f >= _e3833_);
                                                    }
                                                    let _e3837_ = phi_25216_;
                                                    let _e3838_ = select(_e3833_, 0.000001f, _e3837_);
                                                    let _e3841_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3838_)));
                                                    let _e3845_ = (_e3841_ + (_e3841_ * (1f - (_e3838_ * _e3841_))));
                                                    let _e3849_ = (_e3845_ + (_e3845_ * (1f - (_e3838_ * _e3845_))));
                                                    let _e3859_ = (_e3780_ <= 1000000000000000f);
                                                    let _e3869_ = select(select(select(_e3780_, (0.5f * (_e3785_ + (_e3786_ * _e3832_))), _e3859_), select(-0.0005f, -(((0.0005f * (_e3785_ * (_e3849_ + (_e3849_ * (1f - (_e3838_ * _e3849_)))))) * (1f + (_e3785_ * _e3832_)))), _e3859_), (_e3777_ <= 0f)), 0f, (_e3780_ <= 0f));
                                                    if (_e3777_ <= -1f) {
                                                        phi_8989_ = -1f;
                                                    } else {
                                                        phi_8989_ = select(_e3777_, 1f, (_e3777_ >= 1f));
                                                    }
                                                    let _e3874_ = phi_8989_;
                                                    if (_e3705_ <= 0f) {
                                                        phi_9002_ = _e3869_;
                                                    } else {
                                                        phi_9002_ = (((1f - _e3705_) * _e3869_) + (_e3705_ * (_e3874_ * (1f - (0.33333334f * (_e3874_ * _e3874_))))));
                                                    }
                                                    let _e3885_ = phi_9002_;
                                                    let _e3889_ = bitcast<u32>(((1f - (_e3705_ * _e3769_)) * _e3629_));
                                                    let _e3890_ = (_e3889_ & 2139095040u);
                                                    if (_e3890_ == 2139095040u) {
                                                        phi_9016_ = true;
                                                    } else {
                                                        phi_9016_ = (_e3890_ == 0u);
                                                    }
                                                    let _e3894_ = phi_9016_;
                                                    let _e3898_ = bitcast<u32>((_e3769_ * _e3885_));
                                                    let _e3899_ = (_e3898_ & 2139095040u);
                                                    if (_e3899_ == 2139095040u) {
                                                        phi_9033_ = true;
                                                    } else {
                                                        phi_9033_ = (_e3899_ == 0u);
                                                    }
                                                    let _e3903_ = phi_9033_;
                                                    phi_9041_ = type_28_(bitcast<f32>(select(_e3889_, 0u, _e3894_)), bitcast<f32>(select(_e3898_, 0u, _e3903_)));
                                                } else {
                                                    phi_9041_ = type_28_(1f, 0f);
                                                }
                                                let _e3908_ = phi_9041_;
                                                let _e3911_ = ((16u + select(0u, 4u, _e173_)) + _e3570_.member_1_);
                                                if (_e3911_ < 256u) {
                                                } else {
                                                    phi_34710_ = true;
                                                    break;
                                                }
                                                let _e3914_ = aux_scratch[_e3911_];
                                                let _e3917_ = select(0u, 256u, _e173_);
                                                let _e3918_ = (_e170_ & 127u);
                                                let _e3919_ = (_e3917_ + _e3918_);
                                                let _e3920_ = (_e3919_ < 1536u);
                                                if _e3920_ {
                                                } else {
                                                    phi_34710_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e3919_] = _e3908_.member;
                                                let _e3922_ = (_e3917_ + 128u);
                                                let _e3923_ = (_e3922_ + _e3918_);
                                                let _e3924_ = (_e3923_ < 1536u);
                                                if _e3924_ {
                                                } else {
                                                    phi_34710_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e3923_] = _e3908_.member_1_;
                                                workgroupBarrier();
                                                phi_9082_ = 1u;
                                                phi_9085_ = _e3908_;
                                                loop {
                                                    let _e3927_ = phi_9082_;
                                                    let _e3929_ = phi_9085_;
                                                    local_39_ = _e3929_;
                                                    local_40_ = _e3929_;
                                                    let _e3930_ = (_e3927_ < 128u);
                                                    if _e3930_ {
                                                        if (_e170_ >= _e3927_) {
                                                            let _e3933_ = ((_e170_ - _e3927_) & 127u);
                                                            let _e3934_ = (_e3917_ + _e3933_);
                                                            if (_e3934_ < 1536u) {
                                                            } else {
                                                                phi_34559_ = true;
                                                                break;
                                                            }
                                                            let _e3937_ = scan_scratch[_e3934_];
                                                            let _e3938_ = (_e3922_ + _e3933_);
                                                            if (_e3938_ < 1536u) {
                                                            } else {
                                                                phi_34559_ = true;
                                                                break;
                                                            }
                                                            let _e3941_ = scan_scratch[_e3938_];
                                                            phi_9125_ = type_28_((_e3929_.member * _e3937_), ((_e3929_.member * _e3941_) + _e3929_.member_1_));
                                                        } else {
                                                            phi_9125_ = _e3929_;
                                                        }
                                                        let _e3949_ = phi_9125_;
                                                        workgroupBarrier();
                                                        if _e3920_ {
                                                        } else {
                                                            phi_34559_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e3919_] = _e3949_.member;
                                                        if _e3924_ {
                                                        } else {
                                                            phi_34559_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e3923_] = _e3949_.member_1_;
                                                        workgroupBarrier();
                                                        phi_9083_ = (_e3927_ << bitcast<u32>(1i));
                                                        phi_9086_ = _e3949_;
                                                    } else {
                                                        phi_9083_ = u32();
                                                        phi_9086_ = type_28_();
                                                    }
                                                    let _e3955_ = phi_9083_;
                                                    let _e3957_ = phi_9086_;
                                                    phi_9082_ = _e3955_;
                                                    phi_9085_ = _e3957_;
                                                    phi_34559_ = _e3553_;
                                                    if !(_e3930_) {
                                                        break;
                                                    }
                                                }
                                                let _e3960_ = phi_34559_;
                                                phi_34710_ = _e3960_;
                                                if _e3960_ {
                                                    break;
                                                }
                                                let _e3962_ = local_39_;
                                                let _e3965_ = local_40_;
                                                if _e1588_ {
                                                    if (_e1587_ <= 128u) {
                                                        phi_9161_ = (_e1587_ - 1u);
                                                    } else {
                                                        phi_9161_ = 127u;
                                                    }
                                                    let _e3972_ = phi_9161_;
                                                    let _e3973_ = (_e3972_ & 127u);
                                                    let _e3974_ = (_e3917_ + _e3973_);
                                                    if (_e3974_ < 1536u) {
                                                    } else {
                                                        phi_34710_ = true;
                                                        break;
                                                    }
                                                    let _e3977_ = scan_scratch[_e3974_];
                                                    let _e3978_ = (_e3922_ + _e3973_);
                                                    if (_e3978_ < 1536u) {
                                                    } else {
                                                        phi_34710_ = true;
                                                        break;
                                                    }
                                                    let _e3981_ = scan_scratch[_e3978_];
                                                    phi_9182_ = ((_e3977_ * _e3914_) + _e3981_);
                                                } else {
                                                    phi_9182_ = _e3914_;
                                                }
                                                let _e3985_ = phi_9182_;
                                                workgroupBarrier();
                                                if _e1589_ {
                                                    let _e3986_ = bitcast<u32>(((_e3962_.member * _e3914_) + _e3965_.member_1_));
                                                    let _e3987_ = (_e3986_ & 2139095040u);
                                                    if (_e3987_ == 2139095040u) {
                                                        phi_9196_ = true;
                                                    } else {
                                                        phi_9196_ = (_e3987_ == 0u);
                                                    }
                                                    let _e3991_ = phi_9196_;
                                                    let _e3997_ = ((((_e3570_.member_1_ * 4u) + _e3725_) * 128u) + _e3918_);
                                                    if (_e3997_ < 2048u) {
                                                    } else {
                                                        phi_34710_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e3997_] = bitcast<f32>(select(_e3986_, 0u, _e3991_));
                                                }
                                                if _e1190_ {
                                                    let _e4000_ = bitcast<u32>(_e3985_);
                                                    let _e4001_ = (_e4000_ & 2139095040u);
                                                    if (_e4001_ == 2139095040u) {
                                                        phi_9227_ = true;
                                                    } else {
                                                        phi_9227_ = (_e4001_ == 0u);
                                                    }
                                                    let _e4005_ = phi_9227_;
                                                    let _e4008_ = (16u + _e3570_.member_1_);
                                                    if (_e4008_ < 256u) {
                                                    } else {
                                                        phi_34710_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e4008_] = bitcast<f32>(select(_e4000_, 0u, _e4005_));
                                                }
                                                if (_e169_ == 128u) {
                                                    let _e4012_ = bitcast<u32>(_e3985_);
                                                    let _e4013_ = (_e4012_ & 2139095040u);
                                                    if (_e4013_ == 2139095040u) {
                                                        phi_9254_ = true;
                                                    } else {
                                                        phi_9254_ = (_e4013_ == 0u);
                                                    }
                                                    let _e4017_ = phi_9254_;
                                                    let _e4020_ = (20u + _e3570_.member_1_);
                                                    if (_e4020_ < 256u) {
                                                    } else {
                                                        phi_34710_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e4020_] = bitcast<f32>(select(_e4012_, 0u, _e4017_));
                                                }
                                                workgroupBarrier();
                                                phi_34712_ = _e3960_;
                                            } else {
                                                phi_34712_ = _e3553_;
                                            }
                                            let _e4024_ = phi_34712_;
                                            phi_34711_ = _e4024_;
                                        } else {
                                            phi_34711_ = _e3553_;
                                        }
                                        let _e4026_ = phi_34711_;
                                        phi_34585_ = _e4026_;
                                        phi_7861_ = _e3568_;
                                        phi_34710_ = _e4026_;
                                        if !(_e3574_) {
                                            break;
                                        }
                                    }
                                    let _e4029_ = phi_34710_;
                                    phi_36576_ = _e4029_;
                                    if _e4029_ {
                                        break;
                                    }
                                    if _e209_ {
                                        phi_34906_ = _e4029_;
                                        phi_9272_ = type_15_(0u, 4u);
                                        loop {
                                            let _e4409_ = phi_34906_;
                                            let _e4411_ = phi_9272_;
                                            if (_e4411_.member < _e4411_.member_1_) {
                                                phi_9273_ = type_15_((_e4411_.member + 1u), _e4411_.member_1_);
                                                phi_9296_ = type_15_(1u, _e4411_.member);
                                            } else {
                                                phi_9273_ = _e4411_;
                                                phi_9296_ = type_15_(0u, u32());
                                            }
                                            let _e4424_ = phi_9273_;
                                            let _e4426_ = phi_9296_;
                                            let _e4430_ = (bitcast<i32>(_e4426_.member) != 0i);
                                            if _e4430_ {
                                                let _e4431_ = (_e4426_.member_1_ & 3u);
                                                switch bitcast<i32>(_e4431_) {
                                                    case 0: {
                                                        phi_9314_ = _e382_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9314_ = _e382_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9314_ = _e382_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9314_ = _e382_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4438_ = phi_9314_;
                                                switch bitcast<i32>(_e4431_) {
                                                    case 0: {
                                                        phi_9325_ = _e389_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9325_ = _e389_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9325_ = _e389_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9325_ = _e389_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4445_ = phi_9325_;
                                                switch bitcast<i32>(_e4431_) {
                                                    case 0: {
                                                        phi_9336_ = _e396_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9336_ = _e396_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9336_ = _e396_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9336_ = _e396_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4452_ = phi_9336_;
                                                switch bitcast<i32>(_e4431_) {
                                                    case 0: {
                                                        phi_9347_ = _e403_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9347_ = _e403_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9347_ = _e403_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9347_ = _e403_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4459_ = phi_9347_;
                                                switch bitcast<i32>(_e4431_) {
                                                    case 0: {
                                                        phi_9358_ = _e410_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9358_ = _e410_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9358_ = _e410_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9358_ = _e410_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4466_ = phi_9358_;
                                                let _e4467_ = bitcast<u32>(_e4438_);
                                                let _e4468_ = (_e4467_ & 2139095040u);
                                                if (_e4468_ != 2139095040u) {
                                                    if (_e4438_ <= 0f) {
                                                        phi_9472_ = 0f;
                                                    } else {
                                                        if (_e4438_ >= 0.999995f) {
                                                            phi_9471_ = 0.999995f;
                                                        } else {
                                                            if (_e4468_ == 2139095040u) {
                                                                phi_9465_ = true;
                                                            } else {
                                                                phi_9465_ = (_e4468_ == 0u);
                                                            }
                                                            let _e4477_ = phi_9465_;
                                                            phi_9471_ = bitcast<f32>(select(_e4467_, 0u, _e4477_));
                                                        }
                                                        let _e4481_ = phi_9471_;
                                                        phi_9472_ = _e4481_;
                                                    }
                                                    let _e4483_ = phi_9472_;
                                                    phi_9479_ = _e4483_;
                                                } else {
                                                    phi_9479_ = select(0f, 0.999995f, (_e4467_ == 2139095040u));
                                                }
                                                let _e4485_ = phi_9479_;
                                                let _e4486_ = bitcast<u32>(_e4445_);
                                                let _e4487_ = (_e4486_ & 2139095040u);
                                                if (_e4487_ != 2139095040u) {
                                                    if (_e4445_ <= 0f) {
                                                        phi_9511_ = 0f;
                                                    } else {
                                                        if (_e4445_ >= 1f) {
                                                            phi_9510_ = 1f;
                                                        } else {
                                                            if (_e4487_ == 2139095040u) {
                                                                phi_9504_ = true;
                                                            } else {
                                                                phi_9504_ = (_e4487_ == 0u);
                                                            }
                                                            let _e4496_ = phi_9504_;
                                                            phi_9510_ = bitcast<f32>(select(_e4486_, 0u, _e4496_));
                                                        }
                                                        let _e4500_ = phi_9510_;
                                                        phi_9511_ = _e4500_;
                                                    }
                                                    let _e4502_ = phi_9511_;
                                                    phi_9518_ = _e4502_;
                                                } else {
                                                    phi_9518_ = select(0f, 1f, (_e4486_ == 2139095040u));
                                                }
                                                let _e4504_ = phi_9518_;
                                                let _e4505_ = bitcast<u32>(_e4452_);
                                                let _e4506_ = (_e4505_ & 2139095040u);
                                                if (_e4506_ != 2139095040u) {
                                                    if (_e4452_ <= 0f) {
                                                        phi_9550_ = 0f;
                                                    } else {
                                                        if (_e4452_ >= 1f) {
                                                            phi_9549_ = 1f;
                                                        } else {
                                                            if (_e4506_ == 2139095040u) {
                                                                phi_9543_ = true;
                                                            } else {
                                                                phi_9543_ = (_e4506_ == 0u);
                                                            }
                                                            let _e4515_ = phi_9543_;
                                                            phi_9549_ = bitcast<f32>(select(_e4505_, 0u, _e4515_));
                                                        }
                                                        let _e4519_ = phi_9549_;
                                                        phi_9550_ = _e4519_;
                                                    }
                                                    let _e4521_ = phi_9550_;
                                                    phi_9557_ = _e4521_;
                                                } else {
                                                    phi_9557_ = select(0f, 1f, (_e4505_ == 2139095040u));
                                                }
                                                let _e4523_ = phi_9557_;
                                                let _e4524_ = bitcast<u32>(_e4459_);
                                                let _e4525_ = (_e4524_ & 2139095040u);
                                                if (_e4525_ != 2139095040u) {
                                                    if (_e4459_ <= 0f) {
                                                        phi_9589_ = 0f;
                                                    } else {
                                                        if (_e4459_ >= 64f) {
                                                            phi_9588_ = 64f;
                                                        } else {
                                                            if (_e4525_ == 2139095040u) {
                                                                phi_9582_ = true;
                                                            } else {
                                                                phi_9582_ = (_e4525_ == 0u);
                                                            }
                                                            let _e4534_ = phi_9582_;
                                                            phi_9588_ = bitcast<f32>(select(_e4524_, 0u, _e4534_));
                                                        }
                                                        let _e4538_ = phi_9588_;
                                                        phi_9589_ = _e4538_;
                                                    }
                                                    let _e4540_ = phi_9589_;
                                                    phi_9596_ = _e4540_;
                                                } else {
                                                    phi_9596_ = select(0f, 64f, (_e4524_ == 2139095040u));
                                                }
                                                let _e4542_ = phi_9596_;
                                                let _e4543_ = bitcast<u32>(_e4466_);
                                                let _e4544_ = (_e4543_ & 2139095040u);
                                                if (_e4544_ != 2139095040u) {
                                                    if (_e4466_ <= 0f) {
                                                        phi_9628_ = 0f;
                                                    } else {
                                                        if (_e4466_ >= 1f) {
                                                            phi_9627_ = 1f;
                                                        } else {
                                                            if (_e4544_ == 2139095040u) {
                                                                phi_9621_ = true;
                                                            } else {
                                                                phi_9621_ = (_e4544_ == 0u);
                                                            }
                                                            let _e4553_ = phi_9621_;
                                                            phi_9627_ = bitcast<f32>(select(_e4543_, 0u, _e4553_));
                                                        }
                                                        let _e4557_ = phi_9627_;
                                                        phi_9628_ = _e4557_;
                                                    }
                                                    let _e4559_ = phi_9628_;
                                                    phi_9635_ = _e4559_;
                                                } else {
                                                    phi_9635_ = select(0f, 1f, (_e4543_ == 2139095040u));
                                                }
                                                let _e4561_ = phi_9635_;
                                                switch bitcast<i32>(_e4431_) {
                                                    case 0: {
                                                        phi_9745_ = _e424_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9745_ = _e424_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9745_ = _e424_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9745_ = _e424_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4568_ = phi_9745_;
                                                switch bitcast<i32>(_e4431_) {
                                                    case 0: {
                                                        phi_9756_ = _e431_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_9756_ = _e431_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_9756_ = _e431_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_9756_ = _e431_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4575_ = phi_9756_;
                                                if (_e4523_ == 0f) {
                                                    phi_9826_ = true;
                                                } else {
                                                    phi_9826_ = (_e4561_ >= 1f);
                                                }
                                                let _e4579_ = phi_9826_;
                                                if _e4579_ {
                                                    phi_35661_ = _e4409_;
                                                } else {
                                                    if _e1589_ {
                                                        let _e4585_ = ((((_e4426_.member_1_ * 4u) + select(0u, 2u, _e173_)) * 128u) + (_e170_ & 127u));
                                                        if (_e4585_ < 2048u) {
                                                        } else {
                                                            phi_35659_ = true;
                                                            break;
                                                        }
                                                        let _e4588_ = lane_buf[_e4585_];
                                                        phi_9853_ = _e4588_;
                                                    } else {
                                                        phi_9853_ = 0f;
                                                    }
                                                    let _e4590_ = phi_9853_;
                                                    if _e1589_ {
                                                        let _e4596_ = ((((_e4426_.member_1_ * 4u) + select(1u, 3u, _e173_)) * 128u) + (_e170_ & 127u));
                                                        if (_e4596_ < 2048u) {
                                                        } else {
                                                            phi_35659_ = true;
                                                            break;
                                                        }
                                                        let _e4599_ = lane_buf[_e4596_];
                                                        phi_9873_ = _e4599_;
                                                    } else {
                                                        phi_9873_ = 0f;
                                                    }
                                                    let _e4601_ = phi_9873_;
                                                    let _e4604_ = ((16u + select(0u, 4u, _e173_)) + _e4426_.member_1_);
                                                    if (_e4604_ < 256u) {
                                                    } else {
                                                        phi_35659_ = true;
                                                        break;
                                                    }
                                                    let _e4607_ = aux_scratch[_e4604_];
                                                    let _e4610_ = (select(512u, 640u, _e173_) + _e170_);
                                                    let _e4611_ = (_e4610_ < 1536u);
                                                    if _e4611_ {
                                                    } else {
                                                        phi_35659_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e4610_] = _e4607_;
                                                    workgroupBarrier();
                                                    phi_34878_ = _e4409_;
                                                    phi_9904_ = type_15_(0u, 6u);
                                                    phi_9907_ = _e4607_;
                                                    phi_9909_ = false;
                                                    loop {
                                                        let _e4614_ = phi_34878_;
                                                        let _e4616_ = phi_9904_;
                                                        let _e4618_ = phi_9907_;
                                                        let _e4620_ = phi_9909_;
                                                        local_44_ = _e4618_;
                                                        local_45_ = _e4618_;
                                                        local_46_ = _e4618_;
                                                        local_47_ = _e4618_;
                                                        if (_e4616_.member < _e4616_.member_1_) {
                                                            phi_9905_ = type_15_((_e4616_.member + 1u), _e4616_.member_1_);
                                                            phi_9932_ = type_15_(1u, _e4616_.member);
                                                        } else {
                                                            phi_9905_ = _e4616_;
                                                            phi_9932_ = type_15_(0u, u32());
                                                        }
                                                        let _e4633_ = phi_9905_;
                                                        let _e4635_ = phi_9932_;
                                                        let _e4638_ = (bitcast<i32>(_e4635_.member) != 0i);
                                                        if _e4638_ {
                                                            if _e4620_ {
                                                                phi_35322_ = _e4614_;
                                                                phi_13837_ = _e4618_;
                                                                phi_13838_ = _e4620_;
                                                            } else {
                                                                let _e4639_ = (_e170_ == 0u);
                                                                if _e4639_ {
                                                                    phi_9955_ = _e4607_;
                                                                } else {
                                                                    let _e4640_ = (_e4610_ - 1u);
                                                                    if (_e4640_ < 1536u) {
                                                                    } else {
                                                                        phi_35320_ = true;
                                                                        break;
                                                                    }
                                                                    let _e4643_ = scan_scratch[_e4640_];
                                                                    phi_9955_ = _e4643_;
                                                                }
                                                                let _e4645_ = phi_9955_;
                                                                if _e1589_ {
                                                                    let _e4646_ = (_e4575_ * _e4590_);
                                                                    let _e4648_ = bitcast<u32>((_e4568_ * _e4601_));
                                                                    let _e4649_ = (_e4648_ & 2139095040u);
                                                                    if (_e4649_ == 2139095040u) {
                                                                        phi_9983_ = true;
                                                                    } else {
                                                                        phi_9983_ = (_e4649_ == 0u);
                                                                    }
                                                                    let _e4653_ = phi_9983_;
                                                                    let _e4660_ = (_e4542_ <= 0f);
                                                                    let _e4662_ = (1f - _e4504_);
                                                                    let _e4663_ = (_e4662_ + select((_e4542_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e4648_, 0u, _e4653_))) & 2147483647u))), 0f, _e4660_));
                                                                    if (_e4663_ != _e4663_) {
                                                                        phi_25996_ = true;
                                                                    } else {
                                                                        phi_25996_ = (0f >= _e4663_);
                                                                    }
                                                                    let _e4667_ = phi_25996_;
                                                                    let _e4668_ = select(_e4663_, 0f, _e4667_);
                                                                    if (_e4668_ != _e4668_) {
                                                                        phi_26056_ = true;
                                                                    } else {
                                                                        phi_26056_ = (1f <= _e4668_);
                                                                    }
                                                                    let _e4672_ = phi_26056_;
                                                                    let _e4673_ = select(_e4668_, 1f, _e4672_);
                                                                    let _e4674_ = (_e4485_ * _e4645_);
                                                                    let _e4675_ = (1f - _e4561_);
                                                                    let _e4676_ = (_e4523_ * _e4675_);
                                                                    let _e4678_ = (_e4646_ - (_e4676_ * _e4674_));
                                                                    if (_e4678_ != _e4678_) {
                                                                        phi_26116_ = true;
                                                                    } else {
                                                                        phi_26116_ = (-1f >= _e4678_);
                                                                    }
                                                                    let _e4682_ = phi_26116_;
                                                                    let _e4683_ = select(_e4678_, -1f, _e4682_);
                                                                    if (_e4683_ != _e4683_) {
                                                                        phi_26176_ = true;
                                                                    } else {
                                                                        phi_26176_ = (1f <= _e4683_);
                                                                    }
                                                                    let _e4687_ = phi_26176_;
                                                                    let _e4688_ = select(_e4683_, 1f, _e4687_);
                                                                    let _e4695_ = bitcast<f32>((bitcast<u32>(_e4678_) & 2147483647u));
                                                                    if (_e4695_ != _e4695_) {
                                                                        phi_26236_ = true;
                                                                    } else {
                                                                        phi_26236_ = (1000000000000000f <= _e4695_);
                                                                    }
                                                                    let _e4699_ = phi_26236_;
                                                                    let _e4700_ = select(_e4695_, 1000000000000000f, _e4699_);
                                                                    let _e4701_ = (_e4700_ * _e4700_);
                                                                    let _e4702_ = (_e4701_ + 0.0000010000001f);
                                                                    if (_e4702_ != _e4702_) {
                                                                        phi_26296_ = true;
                                                                    } else {
                                                                        phi_26296_ = (0.000000000001f >= _e4702_);
                                                                    }
                                                                    let _e4706_ = phi_26296_;
                                                                    let _e4707_ = select(_e4702_, 0.000000000001f, _e4706_);
                                                                    let _e4708_ = (_e4707_ * 0.5f);
                                                                    let _e4713_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e4707_) >> bitcast<u32>(1i))));
                                                                    let _e4717_ = (_e4713_ * (1.5f - (_e4708_ * (_e4713_ * _e4713_))));
                                                                    let _e4721_ = (_e4717_ * (1.5f - (_e4708_ * (_e4717_ * _e4717_))));
                                                                    let _e4726_ = (_e4702_ * (_e4721_ * (1.5f - (_e4708_ * (_e4721_ * _e4721_)))));
                                                                    let _e4727_ = (_e4726_ + 0.001f);
                                                                    if (_e4727_ != _e4727_) {
                                                                        phi_26356_ = true;
                                                                    } else {
                                                                        phi_26356_ = (0.000001f >= _e4727_);
                                                                    }
                                                                    let _e4731_ = phi_26356_;
                                                                    let _e4732_ = select(_e4727_, 0.000001f, _e4731_);
                                                                    let _e4735_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4732_)));
                                                                    let _e4739_ = (_e4735_ + (_e4735_ * (1f - (_e4732_ * _e4735_))));
                                                                    let _e4743_ = (_e4739_ + (_e4739_ * (1f - (_e4732_ * _e4739_))));
                                                                    let _e4747_ = (_e4743_ + (_e4743_ * (1f - (_e4732_ * _e4743_))));
                                                                    let _e4748_ = (_e4726_ + _e4700_);
                                                                    if (_e4748_ != _e4748_) {
                                                                        phi_26416_ = true;
                                                                    } else {
                                                                        phi_26416_ = (0.000001f >= _e4748_);
                                                                    }
                                                                    let _e4752_ = phi_26416_;
                                                                    let _e4753_ = select(_e4748_, 0.000001f, _e4752_);
                                                                    let _e4756_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4753_)));
                                                                    let _e4760_ = (_e4756_ + (_e4756_ * (1f - (_e4753_ * _e4756_))));
                                                                    let _e4764_ = (_e4760_ + (_e4760_ * (1f - (_e4753_ * _e4760_))));
                                                                    let _e4774_ = (_e4695_ <= 1000000000000000f);
                                                                    let _e4784_ = select(select(select(_e4695_, (0.5f * (_e4700_ + (_e4701_ * _e4747_))), _e4774_), select(-0.0005f, -(((0.0005f * (_e4700_ * (_e4764_ + (_e4764_ * (1f - (_e4753_ * _e4764_)))))) * (1f + (_e4700_ * _e4747_)))), _e4774_), (_e4678_ <= 0f)), 0f, (_e4695_ <= 0f));
                                                                    let _e4788_ = (_e4561_ <= 0f);
                                                                    let _e4795_ = bitcast<u32>((((1f - (_e4561_ * _e4673_)) * _e4674_) + (_e4673_ * select(((_e4675_ * _e4784_) + (_e4561_ * (_e4688_ * (1f - (0.33333334f * (_e4688_ * _e4688_)))))), _e4784_, _e4788_))));
                                                                    let _e4796_ = (_e4795_ & 2139095040u);
                                                                    if (_e4796_ == 2139095040u) {
                                                                        phi_10733_ = true;
                                                                    } else {
                                                                        phi_10733_ = (_e4796_ == 0u);
                                                                    }
                                                                    let _e4800_ = phi_10733_;
                                                                    let _e4803_ = bitcast<u32>(_e4646_);
                                                                    let _e4804_ = (_e4803_ & 2139095040u);
                                                                    if (_e4804_ == 2139095040u) {
                                                                        phi_10781_ = true;
                                                                    } else {
                                                                        phi_10781_ = (_e4804_ == 0u);
                                                                    }
                                                                    let _e4808_ = phi_10781_;
                                                                    let _e4811_ = bitcast<u32>(_e4601_);
                                                                    let _e4812_ = (_e4811_ & 2139095040u);
                                                                    if (_e4812_ == 2139095040u) {
                                                                        phi_10797_ = true;
                                                                    } else {
                                                                        phi_10797_ = (_e4812_ == 0u);
                                                                    }
                                                                    let _e4816_ = phi_10797_;
                                                                    let _e4819_ = bitcast<u32>(_e4645_);
                                                                    let _e4820_ = (_e4819_ & 2139095040u);
                                                                    if (_e4820_ == 2139095040u) {
                                                                        phi_10813_ = true;
                                                                    } else {
                                                                        phi_10813_ = (_e4820_ == 0u);
                                                                    }
                                                                    let _e4824_ = phi_10813_;
                                                                    let _e4828_ = bitcast<u32>((_e4568_ * bitcast<f32>(select(_e4811_, 0u, _e4816_))));
                                                                    let _e4829_ = (_e4828_ & 2139095040u);
                                                                    if (_e4829_ == 2139095040u) {
                                                                        phi_10830_ = true;
                                                                    } else {
                                                                        phi_10830_ = (_e4829_ == 0u);
                                                                    }
                                                                    let _e4833_ = phi_10830_;
                                                                    if _e4660_ {
                                                                        phi_10842_ = 0f;
                                                                    } else {
                                                                        phi_10842_ = (_e4542_ * abs(bitcast<f32>(select(_e4828_, 0u, _e4833_))));
                                                                    }
                                                                    let _e4839_ = phi_10842_;
                                                                    let _e4840_ = (_e4662_ + _e4839_);
                                                                    if (_e4840_ <= 0f) {
                                                                        phi_10854_ = 0f;
                                                                    } else {
                                                                        phi_10854_ = select(_e4840_, 1f, (_e4840_ >= 1f));
                                                                    }
                                                                    let _e4845_ = phi_10854_;
                                                                    let _e4848_ = (bitcast<f32>(select(_e4803_, 0u, _e4808_)) - (_e4676_ * (_e4485_ * bitcast<f32>(select(_e4819_, 0u, _e4824_)))));
                                                                    let _e4851_ = bitcast<f32>((bitcast<u32>(_e4848_) & 2147483647u));
                                                                    if (_e4851_ != _e4851_) {
                                                                        phi_26596_ = true;
                                                                    } else {
                                                                        phi_26596_ = (1000000000000000f <= _e4851_);
                                                                    }
                                                                    let _e4855_ = phi_26596_;
                                                                    let _e4856_ = select(_e4851_, 1000000000000000f, _e4855_);
                                                                    let _e4858_ = ((_e4856_ * _e4856_) + 0.0000010000001f);
                                                                    if (_e4858_ != _e4858_) {
                                                                        phi_26656_ = true;
                                                                    } else {
                                                                        phi_26656_ = (0.000000000001f >= _e4858_);
                                                                    }
                                                                    let _e4862_ = phi_26656_;
                                                                    let _e4863_ = select(_e4858_, 0.000000000001f, _e4862_);
                                                                    let _e4864_ = (_e4863_ * 0.5f);
                                                                    let _e4869_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e4863_) >> bitcast<u32>(1i))));
                                                                    let _e4873_ = (_e4869_ * (1.5f - (_e4864_ * (_e4869_ * _e4869_))));
                                                                    let _e4877_ = (_e4873_ * (1.5f - (_e4864_ * (_e4873_ * _e4873_))));
                                                                    let _e4881_ = (_e4877_ * (1.5f - (_e4864_ * (_e4877_ * _e4877_))));
                                                                    let _e4883_ = ((_e4858_ * _e4881_) + _e4856_);
                                                                    if (_e4883_ != _e4883_) {
                                                                        phi_26776_ = true;
                                                                    } else {
                                                                        phi_26776_ = (0.000001f >= _e4883_);
                                                                    }
                                                                    let _e4887_ = phi_26776_;
                                                                    let _e4888_ = select(_e4883_, 0.000001f, _e4887_);
                                                                    let _e4891_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4888_)));
                                                                    let _e4895_ = (_e4891_ + (_e4891_ * (1f - (_e4888_ * _e4891_))));
                                                                    let _e4899_ = (_e4895_ + (_e4895_ * (1f - (_e4888_ * _e4895_))));
                                                                    let _e4904_ = (_e4851_ <= 1000000000000000f);
                                                                    let _e4915_ = select(select(select(1f, (0.5f * (1f + (_e4856_ * _e4881_))), _e4904_), select(0f, ((0.00000050000006f * _e4881_) * (_e4899_ + (_e4899_ * (1f - (_e4888_ * _e4899_))))), _e4904_), (_e4848_ <= 0f)), 0.5f, (_e4851_ <= 0f));
                                                                    if (_e4915_ != _e4915_) {
                                                                        phi_26836_ = true;
                                                                    } else {
                                                                        phi_26836_ = (0f >= _e4915_);
                                                                    }
                                                                    let _e4919_ = phi_26836_;
                                                                    let _e4920_ = select(_e4915_, 0f, _e4919_);
                                                                    if (_e4920_ != _e4920_) {
                                                                        phi_26896_ = true;
                                                                    } else {
                                                                        phi_26896_ = (1f <= _e4920_);
                                                                    }
                                                                    let _e4924_ = phi_26896_;
                                                                    let _e4925_ = select(_e4920_, 1f, _e4924_);
                                                                    if (abs(_e4848_) < 1f) {
                                                                        phi_11350_ = (1f - (_e4848_ * _e4848_));
                                                                    } else {
                                                                        phi_11350_ = 0f;
                                                                    }
                                                                    let _e4931_ = phi_11350_;
                                                                    if _e4788_ {
                                                                        phi_11358_ = _e4925_;
                                                                    } else {
                                                                        phi_11358_ = ((_e4675_ * _e4925_) + (_e4561_ * _e4931_));
                                                                    }
                                                                    let _e4936_ = phi_11358_;
                                                                    let _e4943_ = (_e4485_ * ((1f - (_e4561_ * _e4845_)) - (((_e4845_ * _e4523_) * _e4675_) * _e4936_)));
                                                                    let _e4944_ = bitcast<u32>(_e4943_);
                                                                    let _e4945_ = (_e4944_ & 2139095040u);
                                                                    let _e4946_ = (_e4945_ == 2139095040u);
                                                                    if _e4946_ {
                                                                        if ((_e4944_ & 2147483647u) > 2139095040u) {
                                                                            phi_11415_ = f32();
                                                                            phi_11416_ = true;
                                                                        } else {
                                                                            let _e4961_ = ((_e4944_ & 2147483648u) == 0u);
                                                                            if _e4961_ {
                                                                                phi_11413_ = 0.999995f;
                                                                            } else {
                                                                                phi_11413_ = f32();
                                                                            }
                                                                            let _e4963_ = phi_11413_;
                                                                            phi_11415_ = _e4963_;
                                                                            phi_11416_ = select(true, false, _e4961_);
                                                                        }
                                                                        let _e4966_ = phi_11415_;
                                                                        let _e4968_ = phi_11416_;
                                                                        if _e4968_ {
                                                                            phi_11436_ = -0.999995f;
                                                                        } else {
                                                                            phi_11436_ = _e4966_;
                                                                        }
                                                                        let _e4970_ = phi_11436_;
                                                                        phi_11495_ = _e4970_;
                                                                    } else {
                                                                        if (_e4943_ < -0.999995f) {
                                                                            phi_11494_ = -0.999995f;
                                                                        } else {
                                                                            if (_e4943_ > 0.999995f) {
                                                                                phi_11493_ = 0.999995f;
                                                                            } else {
                                                                                if _e4946_ {
                                                                                    phi_11487_ = true;
                                                                                } else {
                                                                                    phi_11487_ = (_e4945_ == 0u);
                                                                                }
                                                                                let _e4951_ = phi_11487_;
                                                                                phi_11493_ = bitcast<f32>(select(_e4944_, 0u, _e4951_));
                                                                            }
                                                                            let _e4955_ = phi_11493_;
                                                                            phi_11494_ = _e4955_;
                                                                        }
                                                                        let _e4957_ = phi_11494_;
                                                                        phi_11495_ = _e4957_;
                                                                    }
                                                                    let _e4972_ = phi_11495_;
                                                                    phi_11498_ = type_28_(bitcast<f32>(select(_e4795_, 0u, _e4800_)), _e4972_);
                                                                } else {
                                                                    phi_11498_ = type_28_(0f, 1f);
                                                                }
                                                                let _e4975_ = phi_11498_;
                                                                if _e1589_ {
                                                                    if ((bitcast<u32>(_e4975_.member) & 2139095040u) != 2139095040u) {
                                                                        let _e4983_ = ((bitcast<u32>(_e4618_) & 2139095040u) != 2139095040u);
                                                                        if _e4983_ {
                                                                            let _e4984_ = abs(_e4975_.member);
                                                                            let _e4985_ = abs(_e4618_);
                                                                            let _e4987_ = select(_e4985_, _e4984_, (_e4984_ > _e4985_));
                                                                            let _e4989_ = select(1f, _e4987_, (_e4987_ > 1f));
                                                                            if ((bitcast<u32>(_e4989_) & 2139095040u) != 2139095040u) {
                                                                                let _e4995_ = (_e4989_ <= 0f);
                                                                                if _e4995_ {
                                                                                    phi_11622_ = f32();
                                                                                } else {
                                                                                    if (_e4989_ < 0.00001f) {
                                                                                        let _e5027_ = (select(0.000000000000000000000001f, _e4989_, (_e4989_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                        if (_e5027_ != _e5027_) {
                                                                                            phi_27016_ = true;
                                                                                        } else {
                                                                                            phi_27016_ = (0.000001f >= _e5027_);
                                                                                        }
                                                                                        let _e5031_ = phi_27016_;
                                                                                        let _e5032_ = select(_e5027_, 0.000001f, _e5031_);
                                                                                        let _e5035_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5032_)));
                                                                                        let _e5039_ = (_e5035_ + (_e5035_ * (1f - (_e5032_ * _e5035_))));
                                                                                        let _e5043_ = (_e5039_ + (_e5039_ * (1f - (_e5032_ * _e5039_))));
                                                                                        let _e5049_ = bitcast<u32>(((_e5043_ + (_e5043_ * (1f - (_e5032_ * _e5043_)))) * 18446744000000000000f));
                                                                                        let _e5050_ = (_e5049_ & 2139095040u);
                                                                                        if (_e5050_ == 2139095040u) {
                                                                                            phi_11580_ = true;
                                                                                        } else {
                                                                                            phi_11580_ = (_e5050_ == 0u);
                                                                                        }
                                                                                        let _e5054_ = phi_11580_;
                                                                                        phi_11621_ = bitcast<f32>(select(_e5049_, 0u, _e5054_));
                                                                                    } else {
                                                                                        if (_e4989_ != _e4989_) {
                                                                                            phi_26956_ = true;
                                                                                        } else {
                                                                                            phi_26956_ = (0.000001f >= _e4989_);
                                                                                        }
                                                                                        let _e5000_ = phi_26956_;
                                                                                        let _e5001_ = select(_e4989_, 0.000001f, _e5000_);
                                                                                        let _e5004_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5001_)));
                                                                                        let _e5008_ = (_e5004_ + (_e5004_ * (1f - (_e5001_ * _e5004_))));
                                                                                        let _e5012_ = (_e5008_ + (_e5008_ * (1f - (_e5001_ * _e5008_))));
                                                                                        let _e5017_ = bitcast<u32>((_e5012_ + (_e5012_ * (1f - (_e5001_ * _e5012_)))));
                                                                                        let _e5018_ = (_e5017_ & 2139095040u);
                                                                                        if (_e5018_ == 2139095040u) {
                                                                                            phi_11615_ = true;
                                                                                        } else {
                                                                                            phi_11615_ = (_e5018_ == 0u);
                                                                                        }
                                                                                        let _e5022_ = phi_11615_;
                                                                                        phi_11621_ = bitcast<f32>(select(_e5017_, 0u, _e5022_));
                                                                                    }
                                                                                    let _e5058_ = phi_11621_;
                                                                                    phi_11622_ = _e5058_;
                                                                                }
                                                                                let _e5060_ = phi_11622_;
                                                                                phi_11623_ = _e5060_;
                                                                                phi_11624_ = _e4995_;
                                                                            } else {
                                                                                phi_11623_ = f32();
                                                                                phi_11624_ = true;
                                                                            }
                                                                            let _e5062_ = phi_11623_;
                                                                            let _e5064_ = phi_11624_;
                                                                            phi_11630_ = (abs((_e4975_.member - _e4618_)) * select(_e5062_, 0f, _e5064_));
                                                                        } else {
                                                                            phi_11630_ = f32();
                                                                        }
                                                                        let _e5068_ = phi_11630_;
                                                                        phi_11632_ = _e5068_;
                                                                        phi_11633_ = select(true, false, _e4983_);
                                                                    } else {
                                                                        phi_11632_ = f32();
                                                                        phi_11633_ = true;
                                                                    }
                                                                    let _e5071_ = phi_11632_;
                                                                    let _e5073_ = phi_11633_;
                                                                    phi_11638_ = select(_e5071_, 1000000000f, _e5073_);
                                                                } else {
                                                                    phi_11638_ = 0f;
                                                                }
                                                                let _e5076_ = phi_11638_;
                                                                let _e5077_ = (_e169_ < 128u);
                                                                if _e5077_ {
                                                                    let _e5078_ = (32u + _e170_);
                                                                    if (_e5078_ < 256u) {
                                                                    } else {
                                                                        phi_35320_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e5078_] = _e5076_;
                                                                }
                                                                workgroupBarrier();
                                                                let _e5081_ = (_e169_ >= 128u);
                                                                if _e5081_ {
                                                                    let _e5082_ = (32u + _e170_);
                                                                    let _e5083_ = (_e5082_ < 256u);
                                                                    if _e5083_ {
                                                                    } else {
                                                                        phi_35320_ = true;
                                                                        break;
                                                                    }
                                                                    let _e5085_ = aux_scratch[_e5082_];
                                                                    if _e5083_ {
                                                                    } else {
                                                                        phi_35320_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e5082_] = select(_e5085_, _e5076_, (_e5076_ > _e5085_));
                                                                }
                                                                workgroupBarrier();
                                                                phi_11676_ = 64u;
                                                                loop {
                                                                    let _e5089_ = phi_11676_;
                                                                    let _e5090_ = (_e5089_ > 0u);
                                                                    if _e5090_ {
                                                                        if (_e170_ < _e5089_) {
                                                                            if _e5077_ {
                                                                                let _e5092_ = (32u + _e170_);
                                                                                let _e5093_ = (_e5092_ < 256u);
                                                                                if _e5093_ {
                                                                                } else {
                                                                                    phi_34869_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e5095_ = aux_scratch[_e5092_];
                                                                                let _e5096_ = (_e5092_ + _e5089_);
                                                                                if (_e5096_ < 256u) {
                                                                                } else {
                                                                                    phi_34869_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e5099_ = aux_scratch[_e5096_];
                                                                                if _e5093_ {
                                                                                } else {
                                                                                    phi_34869_ = true;
                                                                                    break;
                                                                                }
                                                                                aux_scratch[_e5092_] = select(_e5095_, _e5099_, (_e5099_ > _e5095_));
                                                                            }
                                                                        }
                                                                        workgroupBarrier();
                                                                        phi_11677_ = (_e5089_ >> bitcast<u32>(1i));
                                                                    } else {
                                                                        phi_11677_ = u32();
                                                                    }
                                                                    let _e5105_ = phi_11677_;
                                                                    phi_11676_ = _e5105_;
                                                                    phi_34869_ = _e4614_;
                                                                    if !(_e5090_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e5108_ = phi_34869_;
                                                                phi_35320_ = _e5108_;
                                                                if _e5108_ {
                                                                    break;
                                                                }
                                                                let _e5110_ = aux_scratch[32];
                                                                workgroupBarrier();
                                                                if (_e5110_ <= 0.0000005f) {
                                                                    phi_35323_ = _e5108_;
                                                                    phi_13835_ = _e4618_;
                                                                    phi_13836_ = true;
                                                                } else {
                                                                    if _e1589_ {
                                                                        phi_11736_ = type_28_(_e4975_.member_1_, (_e4975_.member - (_e4975_.member_1_ * _e4645_)));
                                                                    } else {
                                                                        phi_11736_ = type_28_(1f, 0f);
                                                                    }
                                                                    let _e5116_ = phi_11736_;
                                                                    let _e5119_ = select(0u, 256u, _e173_);
                                                                    let _e5120_ = (_e170_ & 127u);
                                                                    let _e5121_ = (_e5119_ + _e5120_);
                                                                    let _e5122_ = (_e5121_ < 1536u);
                                                                    if _e5122_ {
                                                                    } else {
                                                                        phi_35320_ = true;
                                                                        break;
                                                                    }
                                                                    scan_scratch[_e5121_] = _e5116_.member;
                                                                    let _e5124_ = (_e5119_ + 128u);
                                                                    let _e5125_ = (_e5124_ + _e5120_);
                                                                    let _e5126_ = (_e5125_ < 1536u);
                                                                    if _e5126_ {
                                                                    } else {
                                                                        phi_35320_ = true;
                                                                        break;
                                                                    }
                                                                    scan_scratch[_e5125_] = _e5116_.member_1_;
                                                                    workgroupBarrier();
                                                                    phi_11764_ = 1u;
                                                                    phi_11767_ = _e5116_;
                                                                    loop {
                                                                        let _e5129_ = phi_11764_;
                                                                        let _e5131_ = phi_11767_;
                                                                        local_42_ = _e5131_;
                                                                        local_43_ = _e5131_;
                                                                        let _e5132_ = (_e5129_ < 128u);
                                                                        if _e5132_ {
                                                                            if (_e170_ >= _e5129_) {
                                                                                let _e5135_ = ((_e170_ - _e5129_) & 127u);
                                                                                let _e5136_ = (_e5119_ + _e5135_);
                                                                                if (_e5136_ < 1536u) {
                                                                                } else {
                                                                                    phi_35160_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e5139_ = scan_scratch[_e5136_];
                                                                                let _e5140_ = (_e5124_ + _e5135_);
                                                                                if (_e5140_ < 1536u) {
                                                                                } else {
                                                                                    phi_35160_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e5143_ = scan_scratch[_e5140_];
                                                                                phi_11807_ = type_28_((_e5131_.member * _e5139_), ((_e5131_.member * _e5143_) + _e5131_.member_1_));
                                                                            } else {
                                                                                phi_11807_ = _e5131_;
                                                                            }
                                                                            let _e5151_ = phi_11807_;
                                                                            workgroupBarrier();
                                                                            if _e5122_ {
                                                                            } else {
                                                                                phi_35160_ = true;
                                                                                break;
                                                                            }
                                                                            scan_scratch[_e5121_] = _e5151_.member;
                                                                            if _e5126_ {
                                                                            } else {
                                                                                phi_35160_ = true;
                                                                                break;
                                                                            }
                                                                            scan_scratch[_e5125_] = _e5151_.member_1_;
                                                                            workgroupBarrier();
                                                                            phi_11765_ = (_e5129_ << bitcast<u32>(1i));
                                                                            phi_11768_ = _e5151_;
                                                                        } else {
                                                                            phi_11765_ = u32();
                                                                            phi_11768_ = type_28_();
                                                                        }
                                                                        let _e5157_ = phi_11765_;
                                                                        let _e5159_ = phi_11768_;
                                                                        phi_11764_ = _e5157_;
                                                                        phi_11767_ = _e5159_;
                                                                        phi_35160_ = _e5108_;
                                                                        if !(_e5132_) {
                                                                            break;
                                                                        }
                                                                    }
                                                                    let _e5162_ = phi_35160_;
                                                                    phi_35320_ = _e5162_;
                                                                    if _e5162_ {
                                                                        break;
                                                                    }
                                                                    let _e5164_ = local_42_;
                                                                    let _e5167_ = local_43_;
                                                                    if _e1588_ {
                                                                        if (_e1587_ <= 128u) {
                                                                            phi_11843_ = (_e1587_ - 1u);
                                                                        } else {
                                                                            phi_11843_ = 127u;
                                                                        }
                                                                        let _e5174_ = phi_11843_;
                                                                        let _e5175_ = (_e5174_ & 127u);
                                                                        if ((_e5119_ + _e5175_) < 1536u) {
                                                                        } else {
                                                                            phi_35320_ = true;
                                                                            break;
                                                                        }
                                                                        if ((_e5124_ + _e5175_) < 1536u) {
                                                                        } else {
                                                                            phi_35320_ = true;
                                                                            break;
                                                                        }
                                                                    }
                                                                    workgroupBarrier();
                                                                    phi_35175_ = _e5162_;
                                                                    phi_11865_ = type_34_(0i, 3i);
                                                                    phi_11868_ = _e4618_;
                                                                    phi_11870_ = 1f;
                                                                    phi_11872_ = false;
                                                                    phi_11874_ = _e4620_;
                                                                    loop {
                                                                        let _e5182_ = phi_35175_;
                                                                        let _e5184_ = phi_11865_;
                                                                        let _e5186_ = phi_11868_;
                                                                        let _e5188_ = phi_11870_;
                                                                        let _e5190_ = phi_11872_;
                                                                        let _e5192_ = phi_11874_;
                                                                        local_51_ = _e5186_;
                                                                        local_52_ = _e5192_;
                                                                        if (_e5184_.member < _e5184_.member_1_) {
                                                                            let _e5199_ = (_e5184_.member + 1i);
                                                                            if select(false, true, ((false == (_e5199_ > _e5184_.member)) != false)) {
                                                                                phi_11901_ = type_35_(0u, i32());
                                                                            } else {
                                                                                phi_11901_ = type_35_(1u, _e5199_);
                                                                            }
                                                                            let _e5209_ = phi_11901_;
                                                                            if (bitcast<i32>(_e5209_.member) != 0i) {
                                                                            } else {
                                                                                phi_35314_ = true;
                                                                                break;
                                                                            }
                                                                            phi_11866_ = type_34_(_e5209_.member_1_, _e5184_.member_1_);
                                                                            phi_11915_ = type_35_(1u, _e5184_.member);
                                                                        } else {
                                                                            phi_11866_ = _e5184_;
                                                                            phi_11915_ = type_35_(0u, i32());
                                                                        }
                                                                        let _e5219_ = phi_11866_;
                                                                        let _e5221_ = phi_11915_;
                                                                        let _e5225_ = (bitcast<i32>(_e5221_.member) != 0i);
                                                                        if _e5225_ {
                                                                            if _e5190_ {
                                                                                phi_35316_ = _e5182_;
                                                                                phi_13831_ = _e5186_;
                                                                                phi_13832_ = _e5188_;
                                                                                phi_13833_ = _e5190_;
                                                                                phi_13834_ = _e5192_;
                                                                            } else {
                                                                                if _e1589_ {
                                                                                    let _e5227_ = (_e5186_ + (_e5188_ * (((_e5164_.member * _e4607_) + _e5167_.member_1_) - _e4618_)));
                                                                                    let _e5228_ = bitcast<u32>(_e5227_);
                                                                                    let _e5229_ = (_e5228_ & 2139095040u);
                                                                                    let _e5230_ = (_e5229_ == 2139095040u);
                                                                                    if _e5230_ {
                                                                                        if ((_e5228_ & 2147483647u) > 2139095040u) {
                                                                                            phi_11975_ = f32();
                                                                                            phi_11976_ = true;
                                                                                        } else {
                                                                                            let _e5245_ = ((_e5228_ & 2147483648u) == 0u);
                                                                                            if _e5245_ {
                                                                                                phi_11973_ = 1000000f;
                                                                                            } else {
                                                                                                phi_11973_ = f32();
                                                                                            }
                                                                                            let _e5247_ = phi_11973_;
                                                                                            phi_11975_ = _e5247_;
                                                                                            phi_11976_ = select(true, false, _e5245_);
                                                                                        }
                                                                                        let _e5250_ = phi_11975_;
                                                                                        let _e5252_ = phi_11976_;
                                                                                        if _e5252_ {
                                                                                            phi_11996_ = -1000000f;
                                                                                        } else {
                                                                                            phi_11996_ = _e5250_;
                                                                                        }
                                                                                        let _e5254_ = phi_11996_;
                                                                                        phi_12055_ = _e5254_;
                                                                                    } else {
                                                                                        if (_e5227_ < -1000000f) {
                                                                                            phi_12054_ = -1000000f;
                                                                                        } else {
                                                                                            if (_e5227_ > 1000000f) {
                                                                                                phi_12053_ = 1000000f;
                                                                                            } else {
                                                                                                if _e5230_ {
                                                                                                    phi_12047_ = true;
                                                                                                } else {
                                                                                                    phi_12047_ = (_e5229_ == 0u);
                                                                                                }
                                                                                                let _e5235_ = phi_12047_;
                                                                                                phi_12053_ = bitcast<f32>(select(_e5228_, 0u, _e5235_));
                                                                                            }
                                                                                            let _e5239_ = phi_12053_;
                                                                                            phi_12054_ = _e5239_;
                                                                                        }
                                                                                        let _e5241_ = phi_12054_;
                                                                                        phi_12055_ = _e5241_;
                                                                                    }
                                                                                    let _e5256_ = phi_12055_;
                                                                                    phi_12056_ = _e5256_;
                                                                                } else {
                                                                                    phi_12056_ = _e4607_;
                                                                                }
                                                                                let _e5258_ = phi_12056_;
                                                                                let _e5259_ = (select(768u, 896u, _e173_) + _e170_);
                                                                                if (_e5259_ < 1536u) {
                                                                                } else {
                                                                                    phi_35314_ = true;
                                                                                    break;
                                                                                }
                                                                                scan_scratch[_e5259_] = _e5258_;
                                                                                workgroupBarrier();
                                                                                if _e4639_ {
                                                                                    phi_12077_ = _e4607_;
                                                                                } else {
                                                                                    let _e5262_ = (_e5259_ - 1u);
                                                                                    if (_e5262_ < 1536u) {
                                                                                    } else {
                                                                                        phi_35314_ = true;
                                                                                        break;
                                                                                    }
                                                                                    let _e5265_ = scan_scratch[_e5262_];
                                                                                    phi_12077_ = _e5265_;
                                                                                }
                                                                                let _e5267_ = phi_12077_;
                                                                                if _e1589_ {
                                                                                    let _e5270_ = bitcast<u32>((_e4568_ * _e4601_));
                                                                                    let _e5271_ = (_e5270_ & 2139095040u);
                                                                                    if (_e5271_ == 2139095040u) {
                                                                                        phi_12105_ = true;
                                                                                    } else {
                                                                                        phi_12105_ = (_e5271_ == 0u);
                                                                                    }
                                                                                    let _e5275_ = phi_12105_;
                                                                                    let _e5285_ = ((1f - _e4504_) + select((_e4542_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e5270_, 0u, _e5275_))) & 2147483647u))), 0f, (_e4542_ <= 0f)));
                                                                                    if (_e5285_ != _e5285_) {
                                                                                        phi_27076_ = true;
                                                                                    } else {
                                                                                        phi_27076_ = (0f >= _e5285_);
                                                                                    }
                                                                                    let _e5289_ = phi_27076_;
                                                                                    let _e5290_ = select(_e5285_, 0f, _e5289_);
                                                                                    if (_e5290_ != _e5290_) {
                                                                                        phi_27136_ = true;
                                                                                    } else {
                                                                                        phi_27136_ = (1f <= _e5290_);
                                                                                    }
                                                                                    let _e5294_ = phi_27136_;
                                                                                    let _e5295_ = select(_e5290_, 1f, _e5294_);
                                                                                    let _e5296_ = (_e4485_ * _e5267_);
                                                                                    let _e5297_ = (1f - _e4561_);
                                                                                    let _e5300_ = ((_e4575_ * _e4590_) - ((_e4523_ * _e5297_) * _e5296_));
                                                                                    if (_e5300_ != _e5300_) {
                                                                                        phi_27196_ = true;
                                                                                    } else {
                                                                                        phi_27196_ = (-1f >= _e5300_);
                                                                                    }
                                                                                    let _e5304_ = phi_27196_;
                                                                                    let _e5305_ = select(_e5300_, -1f, _e5304_);
                                                                                    if (_e5305_ != _e5305_) {
                                                                                        phi_27256_ = true;
                                                                                    } else {
                                                                                        phi_27256_ = (1f <= _e5305_);
                                                                                    }
                                                                                    let _e5309_ = phi_27256_;
                                                                                    let _e5310_ = select(_e5305_, 1f, _e5309_);
                                                                                    let _e5317_ = bitcast<f32>((bitcast<u32>(_e5300_) & 2147483647u));
                                                                                    if (_e5317_ != _e5317_) {
                                                                                        phi_27316_ = true;
                                                                                    } else {
                                                                                        phi_27316_ = (1000000000000000f <= _e5317_);
                                                                                    }
                                                                                    let _e5321_ = phi_27316_;
                                                                                    let _e5322_ = select(_e5317_, 1000000000000000f, _e5321_);
                                                                                    let _e5323_ = (_e5322_ * _e5322_);
                                                                                    let _e5324_ = (_e5323_ + 0.0000010000001f);
                                                                                    if (_e5324_ != _e5324_) {
                                                                                        phi_27376_ = true;
                                                                                    } else {
                                                                                        phi_27376_ = (0.000000000001f >= _e5324_);
                                                                                    }
                                                                                    let _e5328_ = phi_27376_;
                                                                                    let _e5329_ = select(_e5324_, 0.000000000001f, _e5328_);
                                                                                    let _e5330_ = (_e5329_ * 0.5f);
                                                                                    let _e5335_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e5329_) >> bitcast<u32>(1i))));
                                                                                    let _e5339_ = (_e5335_ * (1.5f - (_e5330_ * (_e5335_ * _e5335_))));
                                                                                    let _e5343_ = (_e5339_ * (1.5f - (_e5330_ * (_e5339_ * _e5339_))));
                                                                                    let _e5348_ = (_e5324_ * (_e5343_ * (1.5f - (_e5330_ * (_e5343_ * _e5343_)))));
                                                                                    let _e5349_ = (_e5348_ + 0.001f);
                                                                                    if (_e5349_ != _e5349_) {
                                                                                        phi_27436_ = true;
                                                                                    } else {
                                                                                        phi_27436_ = (0.000001f >= _e5349_);
                                                                                    }
                                                                                    let _e5353_ = phi_27436_;
                                                                                    let _e5354_ = select(_e5349_, 0.000001f, _e5353_);
                                                                                    let _e5357_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5354_)));
                                                                                    let _e5361_ = (_e5357_ + (_e5357_ * (1f - (_e5354_ * _e5357_))));
                                                                                    let _e5365_ = (_e5361_ + (_e5361_ * (1f - (_e5354_ * _e5361_))));
                                                                                    let _e5369_ = (_e5365_ + (_e5365_ * (1f - (_e5354_ * _e5365_))));
                                                                                    let _e5370_ = (_e5348_ + _e5322_);
                                                                                    if (_e5370_ != _e5370_) {
                                                                                        phi_27496_ = true;
                                                                                    } else {
                                                                                        phi_27496_ = (0.000001f >= _e5370_);
                                                                                    }
                                                                                    let _e5374_ = phi_27496_;
                                                                                    let _e5375_ = select(_e5370_, 0.000001f, _e5374_);
                                                                                    let _e5378_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5375_)));
                                                                                    let _e5382_ = (_e5378_ + (_e5378_ * (1f - (_e5375_ * _e5378_))));
                                                                                    let _e5386_ = (_e5382_ + (_e5382_ * (1f - (_e5375_ * _e5382_))));
                                                                                    let _e5396_ = (_e5317_ <= 1000000000000000f);
                                                                                    let _e5406_ = select(select(select(_e5317_, (0.5f * (_e5322_ + (_e5323_ * _e5369_))), _e5396_), select(-0.0005f, -(((0.0005f * (_e5322_ * (_e5386_ + (_e5386_ * (1f - (_e5375_ * _e5386_)))))) * (1f + (_e5322_ * _e5369_)))), _e5396_), (_e5300_ <= 0f)), 0f, (_e5317_ <= 0f));
                                                                                    let _e5417_ = bitcast<u32>((((1f - (_e4561_ * _e5295_)) * _e5296_) + (_e5295_ * select(((_e5297_ * _e5406_) + (_e4561_ * (_e5310_ * (1f - (0.33333334f * (_e5310_ * _e5310_)))))), _e5406_, (_e4561_ <= 0f)))));
                                                                                    let _e5418_ = (_e5417_ & 2139095040u);
                                                                                    if (_e5418_ == 2139095040u) {
                                                                                        phi_12855_ = true;
                                                                                    } else {
                                                                                        phi_12855_ = (_e5418_ == 0u);
                                                                                    }
                                                                                    let _e5422_ = phi_12855_;
                                                                                    let _e5424_ = bitcast<f32>(select(_e5417_, 0u, _e5422_));
                                                                                    if ((bitcast<u32>(_e5424_) & 2139095040u) != 2139095040u) {
                                                                                        let _e5430_ = ((bitcast<u32>(_e5258_) & 2139095040u) != 2139095040u);
                                                                                        if _e5430_ {
                                                                                            let _e5431_ = abs(_e5424_);
                                                                                            let _e5432_ = abs(_e5258_);
                                                                                            let _e5434_ = select(_e5432_, _e5431_, (_e5431_ > _e5432_));
                                                                                            let _e5436_ = select(1f, _e5434_, (_e5434_ > 1f));
                                                                                            if ((bitcast<u32>(_e5436_) & 2139095040u) != 2139095040u) {
                                                                                                let _e5442_ = (_e5436_ <= 0f);
                                                                                                if _e5442_ {
                                                                                                    phi_13700_ = f32();
                                                                                                } else {
                                                                                                    if (_e5436_ < 0.00001f) {
                                                                                                        let _e5474_ = (select(0.000000000000000000000001f, _e5436_, (_e5436_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                                        if (_e5474_ != _e5474_) {
                                                                                                            phi_28096_ = true;
                                                                                                        } else {
                                                                                                            phi_28096_ = (0.000001f >= _e5474_);
                                                                                                        }
                                                                                                        let _e5478_ = phi_28096_;
                                                                                                        let _e5479_ = select(_e5474_, 0.000001f, _e5478_);
                                                                                                        let _e5482_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5479_)));
                                                                                                        let _e5486_ = (_e5482_ + (_e5482_ * (1f - (_e5479_ * _e5482_))));
                                                                                                        let _e5490_ = (_e5486_ + (_e5486_ * (1f - (_e5479_ * _e5486_))));
                                                                                                        let _e5496_ = bitcast<u32>(((_e5490_ + (_e5490_ * (1f - (_e5479_ * _e5490_)))) * 18446744000000000000f));
                                                                                                        let _e5497_ = (_e5496_ & 2139095040u);
                                                                                                        if (_e5497_ == 2139095040u) {
                                                                                                            phi_13658_ = true;
                                                                                                        } else {
                                                                                                            phi_13658_ = (_e5497_ == 0u);
                                                                                                        }
                                                                                                        let _e5501_ = phi_13658_;
                                                                                                        phi_13699_ = bitcast<f32>(select(_e5496_, 0u, _e5501_));
                                                                                                    } else {
                                                                                                        if (_e5436_ != _e5436_) {
                                                                                                            phi_28036_ = true;
                                                                                                        } else {
                                                                                                            phi_28036_ = (0.000001f >= _e5436_);
                                                                                                        }
                                                                                                        let _e5447_ = phi_28036_;
                                                                                                        let _e5448_ = select(_e5436_, 0.000001f, _e5447_);
                                                                                                        let _e5451_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5448_)));
                                                                                                        let _e5455_ = (_e5451_ + (_e5451_ * (1f - (_e5448_ * _e5451_))));
                                                                                                        let _e5459_ = (_e5455_ + (_e5455_ * (1f - (_e5448_ * _e5455_))));
                                                                                                        let _e5464_ = bitcast<u32>((_e5459_ + (_e5459_ * (1f - (_e5448_ * _e5459_)))));
                                                                                                        let _e5465_ = (_e5464_ & 2139095040u);
                                                                                                        if (_e5465_ == 2139095040u) {
                                                                                                            phi_13693_ = true;
                                                                                                        } else {
                                                                                                            phi_13693_ = (_e5465_ == 0u);
                                                                                                        }
                                                                                                        let _e5469_ = phi_13693_;
                                                                                                        phi_13699_ = bitcast<f32>(select(_e5464_, 0u, _e5469_));
                                                                                                    }
                                                                                                    let _e5505_ = phi_13699_;
                                                                                                    phi_13700_ = _e5505_;
                                                                                                }
                                                                                                let _e5507_ = phi_13700_;
                                                                                                phi_13701_ = _e5507_;
                                                                                                phi_13702_ = _e5442_;
                                                                                            } else {
                                                                                                phi_13701_ = f32();
                                                                                                phi_13702_ = true;
                                                                                            }
                                                                                            let _e5509_ = phi_13701_;
                                                                                            let _e5511_ = phi_13702_;
                                                                                            phi_13708_ = (abs((_e5424_ - _e5258_)) * select(_e5509_, 0f, _e5511_));
                                                                                        } else {
                                                                                            phi_13708_ = f32();
                                                                                        }
                                                                                        let _e5515_ = phi_13708_;
                                                                                        phi_13710_ = _e5515_;
                                                                                        phi_13711_ = select(true, false, _e5430_);
                                                                                    } else {
                                                                                        phi_13710_ = f32();
                                                                                        phi_13711_ = true;
                                                                                    }
                                                                                    let _e5518_ = phi_13710_;
                                                                                    let _e5520_ = phi_13711_;
                                                                                    phi_13716_ = select(_e5518_, 1000000000f, _e5520_);
                                                                                } else {
                                                                                    phi_13716_ = 0f;
                                                                                }
                                                                                let _e5523_ = phi_13716_;
                                                                                if _e5077_ {
                                                                                    let _e5524_ = (32u + _e170_);
                                                                                    if (_e5524_ < 256u) {
                                                                                    } else {
                                                                                        phi_35314_ = true;
                                                                                        break;
                                                                                    }
                                                                                    aux_scratch[_e5524_] = _e5523_;
                                                                                }
                                                                                workgroupBarrier();
                                                                                if _e5081_ {
                                                                                    let _e5527_ = (32u + _e170_);
                                                                                    let _e5528_ = (_e5527_ < 256u);
                                                                                    if _e5528_ {
                                                                                    } else {
                                                                                        phi_35314_ = true;
                                                                                        break;
                                                                                    }
                                                                                    let _e5530_ = aux_scratch[_e5527_];
                                                                                    if _e5528_ {
                                                                                    } else {
                                                                                        phi_35314_ = true;
                                                                                        break;
                                                                                    }
                                                                                    aux_scratch[_e5527_] = select(_e5530_, _e5523_, (_e5523_ > _e5530_));
                                                                                }
                                                                                workgroupBarrier();
                                                                                phi_13754_ = 64u;
                                                                                loop {
                                                                                    let _e5534_ = phi_13754_;
                                                                                    let _e5535_ = (_e5534_ > 0u);
                                                                                    if _e5535_ {
                                                                                        if (_e170_ < _e5534_) {
                                                                                            if _e5077_ {
                                                                                                let _e5537_ = (32u + _e170_);
                                                                                                let _e5538_ = (_e5537_ < 256u);
                                                                                                if _e5538_ {
                                                                                                } else {
                                                                                                    phi_35166_ = true;
                                                                                                    break;
                                                                                                }
                                                                                                let _e5540_ = aux_scratch[_e5537_];
                                                                                                let _e5541_ = (_e5537_ + _e5534_);
                                                                                                if (_e5541_ < 256u) {
                                                                                                } else {
                                                                                                    phi_35166_ = true;
                                                                                                    break;
                                                                                                }
                                                                                                let _e5544_ = aux_scratch[_e5541_];
                                                                                                if _e5538_ {
                                                                                                } else {
                                                                                                    phi_35166_ = true;
                                                                                                    break;
                                                                                                }
                                                                                                aux_scratch[_e5537_] = select(_e5540_, _e5544_, (_e5544_ > _e5540_));
                                                                                            }
                                                                                        }
                                                                                        workgroupBarrier();
                                                                                        phi_13755_ = (_e5534_ >> bitcast<u32>(1i));
                                                                                    } else {
                                                                                        phi_13755_ = u32();
                                                                                    }
                                                                                    let _e5550_ = phi_13755_;
                                                                                    phi_13754_ = _e5550_;
                                                                                    phi_35166_ = _e5182_;
                                                                                    if !(_e5535_) {
                                                                                        break;
                                                                                    }
                                                                                }
                                                                                let _e5553_ = phi_35166_;
                                                                                phi_35314_ = _e5553_;
                                                                                if _e5553_ {
                                                                                    break;
                                                                                }
                                                                                let _e5554_ = aux_scratch[32];
                                                                                workgroupBarrier();
                                                                                if (_e5554_ <= _e5110_) {
                                                                                    phi_13811_ = true;
                                                                                } else {
                                                                                    phi_13811_ = (_e5221_.member_1_ == 2i);
                                                                                }
                                                                                let _e5558_ = phi_13811_;
                                                                                if _e5558_ {
                                                                                    if _e4611_ {
                                                                                    } else {
                                                                                        phi_35314_ = true;
                                                                                        break;
                                                                                    }
                                                                                    scan_scratch[_e4610_] = _e5258_;
                                                                                    workgroupBarrier();
                                                                                    phi_13829_ = select(_e5192_, true, (_e5554_ <= 0.0000005f));
                                                                                } else {
                                                                                    phi_13829_ = _e5192_;
                                                                                }
                                                                                let _e5562_ = phi_13829_;
                                                                                phi_35316_ = _e5553_;
                                                                                phi_13831_ = select(_e5186_, _e5258_, _e5558_);
                                                                                phi_13832_ = (_e5188_ * 0.5f);
                                                                                phi_13833_ = select(_e5190_, true, _e5558_);
                                                                                phi_13834_ = _e5562_;
                                                                            }
                                                                            let _e5567_ = phi_35316_;
                                                                            let _e5569_ = phi_13831_;
                                                                            let _e5571_ = phi_13832_;
                                                                            let _e5573_ = phi_13833_;
                                                                            let _e5575_ = phi_13834_;
                                                                            phi_35315_ = _e5567_;
                                                                            phi_11869_ = _e5569_;
                                                                            phi_11871_ = _e5571_;
                                                                            phi_11873_ = _e5573_;
                                                                            phi_11875_ = _e5575_;
                                                                        } else {
                                                                            phi_35315_ = _e5182_;
                                                                            phi_11869_ = f32();
                                                                            phi_11871_ = f32();
                                                                            phi_11873_ = bool();
                                                                            phi_11875_ = bool();
                                                                        }
                                                                        let _e5577_ = phi_35315_;
                                                                        let _e5579_ = phi_11869_;
                                                                        let _e5581_ = phi_11871_;
                                                                        let _e5583_ = phi_11873_;
                                                                        let _e5585_ = phi_11875_;
                                                                        phi_35175_ = _e5577_;
                                                                        phi_11865_ = _e5219_;
                                                                        phi_11868_ = _e5579_;
                                                                        phi_11870_ = _e5581_;
                                                                        phi_11872_ = _e5583_;
                                                                        phi_11874_ = _e5585_;
                                                                        phi_35314_ = _e5577_;
                                                                        if !(_e5225_) {
                                                                            break;
                                                                        }
                                                                    }
                                                                    let _e5588_ = phi_35314_;
                                                                    phi_35320_ = _e5588_;
                                                                    if _e5588_ {
                                                                        break;
                                                                    }
                                                                    phi_35323_ = _e5588_;
                                                                    let _e10462_ = local_51_;
                                                                    phi_13835_ = _e10462_;
                                                                    let _e10465_ = local_52_;
                                                                    phi_13836_ = _e10465_;
                                                                }
                                                                let _e5590_ = phi_35323_;
                                                                let _e5592_ = phi_13835_;
                                                                let _e5594_ = phi_13836_;
                                                                phi_35322_ = _e5590_;
                                                                phi_13837_ = _e5592_;
                                                                phi_13838_ = _e5594_;
                                                            }
                                                            let _e5596_ = phi_35322_;
                                                            let _e5598_ = phi_13837_;
                                                            let _e5600_ = phi_13838_;
                                                            phi_35321_ = _e5596_;
                                                            phi_9908_ = _e5598_;
                                                            phi_9910_ = _e5600_;
                                                        } else {
                                                            phi_35321_ = _e4614_;
                                                            phi_9908_ = f32();
                                                            phi_9910_ = bool();
                                                        }
                                                        let _e5602_ = phi_35321_;
                                                        let _e5604_ = phi_9908_;
                                                        let _e5606_ = phi_9910_;
                                                        phi_34878_ = _e5602_;
                                                        phi_9904_ = _e4633_;
                                                        phi_9907_ = _e5604_;
                                                        phi_9909_ = _e5606_;
                                                        phi_35320_ = _e5602_;
                                                        if !(_e4638_) {
                                                            break;
                                                        }
                                                    }
                                                    let _e5609_ = phi_35320_;
                                                    phi_35659_ = _e5609_;
                                                    if _e5609_ {
                                                        break;
                                                    }
                                                    if (_e170_ == 0u) {
                                                        phi_13852_ = _e4607_;
                                                    } else {
                                                        let _e5611_ = (_e4610_ - 1u);
                                                        if (_e5611_ < 1536u) {
                                                        } else {
                                                            phi_35659_ = true;
                                                            break;
                                                        }
                                                        let _e5614_ = scan_scratch[_e5611_];
                                                        phi_13852_ = _e5614_;
                                                    }
                                                    let _e5616_ = phi_13852_;
                                                    if _e1589_ {
                                                        let _e5620_ = (_e4575_ * _e4590_);
                                                        let _e5622_ = bitcast<u32>((_e4568_ * _e4601_));
                                                        let _e5623_ = (_e5622_ & 2139095040u);
                                                        if (_e5623_ == 2139095040u) {
                                                            phi_13880_ = true;
                                                        } else {
                                                            phi_13880_ = (_e5623_ == 0u);
                                                        }
                                                        let _e5627_ = phi_13880_;
                                                        let _e5634_ = (_e4542_ <= 0f);
                                                        let _e5636_ = (1f - _e4504_);
                                                        let _e5637_ = (_e5636_ + select((_e4542_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e5622_, 0u, _e5627_))) & 2147483647u))), 0f, _e5634_));
                                                        if (_e5637_ != _e5637_) {
                                                            phi_28156_ = true;
                                                        } else {
                                                            phi_28156_ = (0f >= _e5637_);
                                                        }
                                                        let _e5641_ = phi_28156_;
                                                        let _e5642_ = select(_e5637_, 0f, _e5641_);
                                                        if (_e5642_ != _e5642_) {
                                                            phi_28216_ = true;
                                                        } else {
                                                            phi_28216_ = (1f <= _e5642_);
                                                        }
                                                        let _e5646_ = phi_28216_;
                                                        let _e5647_ = select(_e5642_, 1f, _e5646_);
                                                        let _e5648_ = (_e4485_ * _e5616_);
                                                        let _e5649_ = (1f - _e4561_);
                                                        let _e5650_ = (_e4523_ * _e5649_);
                                                        let _e5652_ = (_e5620_ - (_e5650_ * _e5648_));
                                                        if (_e5652_ != _e5652_) {
                                                            phi_28276_ = true;
                                                        } else {
                                                            phi_28276_ = (-1f >= _e5652_);
                                                        }
                                                        let _e5656_ = phi_28276_;
                                                        let _e5657_ = select(_e5652_, -1f, _e5656_);
                                                        if (_e5657_ != _e5657_) {
                                                            phi_28336_ = true;
                                                        } else {
                                                            phi_28336_ = (1f <= _e5657_);
                                                        }
                                                        let _e5661_ = phi_28336_;
                                                        let _e5662_ = select(_e5657_, 1f, _e5661_);
                                                        let _e5669_ = bitcast<f32>((bitcast<u32>(_e5652_) & 2147483647u));
                                                        if (_e5669_ != _e5669_) {
                                                            phi_28396_ = true;
                                                        } else {
                                                            phi_28396_ = (1000000000000000f <= _e5669_);
                                                        }
                                                        let _e5673_ = phi_28396_;
                                                        let _e5674_ = select(_e5669_, 1000000000000000f, _e5673_);
                                                        let _e5675_ = (_e5674_ * _e5674_);
                                                        let _e5676_ = (_e5675_ + 0.0000010000001f);
                                                        if (_e5676_ != _e5676_) {
                                                            phi_28456_ = true;
                                                        } else {
                                                            phi_28456_ = (0.000000000001f >= _e5676_);
                                                        }
                                                        let _e5680_ = phi_28456_;
                                                        let _e5681_ = select(_e5676_, 0.000000000001f, _e5680_);
                                                        let _e5682_ = (_e5681_ * 0.5f);
                                                        let _e5687_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e5681_) >> bitcast<u32>(1i))));
                                                        let _e5691_ = (_e5687_ * (1.5f - (_e5682_ * (_e5687_ * _e5687_))));
                                                        let _e5695_ = (_e5691_ * (1.5f - (_e5682_ * (_e5691_ * _e5691_))));
                                                        let _e5700_ = (_e5676_ * (_e5695_ * (1.5f - (_e5682_ * (_e5695_ * _e5695_)))));
                                                        let _e5701_ = (_e5700_ + 0.001f);
                                                        if (_e5701_ != _e5701_) {
                                                            phi_28516_ = true;
                                                        } else {
                                                            phi_28516_ = (0.000001f >= _e5701_);
                                                        }
                                                        let _e5705_ = phi_28516_;
                                                        let _e5706_ = select(_e5701_, 0.000001f, _e5705_);
                                                        let _e5709_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5706_)));
                                                        let _e5713_ = (_e5709_ + (_e5709_ * (1f - (_e5706_ * _e5709_))));
                                                        let _e5717_ = (_e5713_ + (_e5713_ * (1f - (_e5706_ * _e5713_))));
                                                        let _e5721_ = (_e5717_ + (_e5717_ * (1f - (_e5706_ * _e5717_))));
                                                        let _e5722_ = (_e5700_ + _e5674_);
                                                        if (_e5722_ != _e5722_) {
                                                            phi_28576_ = true;
                                                        } else {
                                                            phi_28576_ = (0.000001f >= _e5722_);
                                                        }
                                                        let _e5726_ = phi_28576_;
                                                        let _e5727_ = select(_e5722_, 0.000001f, _e5726_);
                                                        let _e5730_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5727_)));
                                                        let _e5734_ = (_e5730_ + (_e5730_ * (1f - (_e5727_ * _e5730_))));
                                                        let _e5738_ = (_e5734_ + (_e5734_ * (1f - (_e5727_ * _e5734_))));
                                                        let _e5748_ = (_e5669_ <= 1000000000000000f);
                                                        let _e5758_ = select(select(select(_e5669_, (0.5f * (_e5674_ + (_e5675_ * _e5721_))), _e5748_), select(-0.0005f, -(((0.0005f * (_e5674_ * (_e5738_ + (_e5738_ * (1f - (_e5727_ * _e5738_)))))) * (1f + (_e5674_ * _e5721_)))), _e5748_), (_e5652_ <= 0f)), 0f, (_e5669_ <= 0f));
                                                        let _e5762_ = (_e4561_ <= 0f);
                                                        let _e5769_ = bitcast<u32>((((1f - (_e4561_ * _e5647_)) * _e5648_) + (_e5647_ * select(((_e5649_ * _e5758_) + (_e4561_ * (_e5662_ * (1f - (0.33333334f * (_e5662_ * _e5662_)))))), _e5758_, _e5762_))));
                                                        let _e5770_ = (_e5769_ & 2139095040u);
                                                        if (_e5770_ == 2139095040u) {
                                                            phi_14630_ = true;
                                                        } else {
                                                            phi_14630_ = (_e5770_ == 0u);
                                                        }
                                                        let _e5774_ = phi_14630_;
                                                        let _e5777_ = bitcast<u32>(_e5620_);
                                                        let _e5778_ = (_e5777_ & 2139095040u);
                                                        if (_e5778_ == 2139095040u) {
                                                            phi_14678_ = true;
                                                        } else {
                                                            phi_14678_ = (_e5778_ == 0u);
                                                        }
                                                        let _e5782_ = phi_14678_;
                                                        let _e5785_ = bitcast<u32>(_e4601_);
                                                        let _e5786_ = (_e5785_ & 2139095040u);
                                                        if (_e5786_ == 2139095040u) {
                                                            phi_14694_ = true;
                                                        } else {
                                                            phi_14694_ = (_e5786_ == 0u);
                                                        }
                                                        let _e5790_ = phi_14694_;
                                                        let _e5793_ = bitcast<u32>(_e5616_);
                                                        let _e5794_ = (_e5793_ & 2139095040u);
                                                        if (_e5794_ == 2139095040u) {
                                                            phi_14710_ = true;
                                                        } else {
                                                            phi_14710_ = (_e5794_ == 0u);
                                                        }
                                                        let _e5798_ = phi_14710_;
                                                        let _e5802_ = bitcast<u32>((_e4568_ * bitcast<f32>(select(_e5785_, 0u, _e5790_))));
                                                        let _e5803_ = (_e5802_ & 2139095040u);
                                                        if (_e5803_ == 2139095040u) {
                                                            phi_14727_ = true;
                                                        } else {
                                                            phi_14727_ = (_e5803_ == 0u);
                                                        }
                                                        let _e5807_ = phi_14727_;
                                                        if _e5634_ {
                                                            phi_14739_ = 0f;
                                                        } else {
                                                            phi_14739_ = (_e4542_ * abs(bitcast<f32>(select(_e5802_, 0u, _e5807_))));
                                                        }
                                                        let _e5813_ = phi_14739_;
                                                        let _e5814_ = (_e5636_ + _e5813_);
                                                        if (_e5814_ <= 0f) {
                                                            phi_14751_ = 0f;
                                                        } else {
                                                            phi_14751_ = select(_e5814_, 1f, (_e5814_ >= 1f));
                                                        }
                                                        let _e5819_ = phi_14751_;
                                                        let _e5822_ = (bitcast<f32>(select(_e5777_, 0u, _e5782_)) - (_e5650_ * (_e4485_ * bitcast<f32>(select(_e5793_, 0u, _e5798_)))));
                                                        let _e5825_ = bitcast<f32>((bitcast<u32>(_e5822_) & 2147483647u));
                                                        if (_e5825_ != _e5825_) {
                                                            phi_28756_ = true;
                                                        } else {
                                                            phi_28756_ = (1000000000000000f <= _e5825_);
                                                        }
                                                        let _e5829_ = phi_28756_;
                                                        let _e5830_ = select(_e5825_, 1000000000000000f, _e5829_);
                                                        let _e5832_ = ((_e5830_ * _e5830_) + 0.0000010000001f);
                                                        if (_e5832_ != _e5832_) {
                                                            phi_28816_ = true;
                                                        } else {
                                                            phi_28816_ = (0.000000000001f >= _e5832_);
                                                        }
                                                        let _e5836_ = phi_28816_;
                                                        let _e5837_ = select(_e5832_, 0.000000000001f, _e5836_);
                                                        let _e5838_ = (_e5837_ * 0.5f);
                                                        let _e5843_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e5837_) >> bitcast<u32>(1i))));
                                                        let _e5847_ = (_e5843_ * (1.5f - (_e5838_ * (_e5843_ * _e5843_))));
                                                        let _e5851_ = (_e5847_ * (1.5f - (_e5838_ * (_e5847_ * _e5847_))));
                                                        let _e5855_ = (_e5851_ * (1.5f - (_e5838_ * (_e5851_ * _e5851_))));
                                                        let _e5857_ = ((_e5832_ * _e5855_) + _e5830_);
                                                        if (_e5857_ != _e5857_) {
                                                            phi_28936_ = true;
                                                        } else {
                                                            phi_28936_ = (0.000001f >= _e5857_);
                                                        }
                                                        let _e5861_ = phi_28936_;
                                                        let _e5862_ = select(_e5857_, 0.000001f, _e5861_);
                                                        let _e5865_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5862_)));
                                                        let _e5869_ = (_e5865_ + (_e5865_ * (1f - (_e5862_ * _e5865_))));
                                                        let _e5873_ = (_e5869_ + (_e5869_ * (1f - (_e5862_ * _e5869_))));
                                                        let _e5878_ = (_e5825_ <= 1000000000000000f);
                                                        let _e5889_ = select(select(select(1f, (0.5f * (1f + (_e5830_ * _e5855_))), _e5878_), select(0f, ((0.00000050000006f * _e5855_) * (_e5873_ + (_e5873_ * (1f - (_e5862_ * _e5873_))))), _e5878_), (_e5822_ <= 0f)), 0.5f, (_e5825_ <= 0f));
                                                        if (_e5889_ != _e5889_) {
                                                            phi_28996_ = true;
                                                        } else {
                                                            phi_28996_ = (0f >= _e5889_);
                                                        }
                                                        let _e5893_ = phi_28996_;
                                                        let _e5894_ = select(_e5889_, 0f, _e5893_);
                                                        if (_e5894_ != _e5894_) {
                                                            phi_29056_ = true;
                                                        } else {
                                                            phi_29056_ = (1f <= _e5894_);
                                                        }
                                                        let _e5898_ = phi_29056_;
                                                        let _e5899_ = select(_e5894_, 1f, _e5898_);
                                                        if (abs(_e5822_) < 1f) {
                                                            phi_15247_ = (1f - (_e5822_ * _e5822_));
                                                        } else {
                                                            phi_15247_ = 0f;
                                                        }
                                                        let _e5905_ = phi_15247_;
                                                        if _e5762_ {
                                                            phi_15255_ = _e5899_;
                                                        } else {
                                                            phi_15255_ = ((_e5649_ * _e5899_) + (_e4561_ * _e5905_));
                                                        }
                                                        let _e5910_ = phi_15255_;
                                                        let _e5917_ = (_e4485_ * ((1f - (_e4561_ * _e5819_)) - (((_e5819_ * _e4523_) * _e5649_) * _e5910_)));
                                                        let _e5918_ = bitcast<u32>(_e5917_);
                                                        let _e5919_ = (_e5918_ & 2139095040u);
                                                        let _e5920_ = (_e5919_ == 2139095040u);
                                                        if _e5920_ {
                                                            if ((_e5918_ & 2147483647u) > 2139095040u) {
                                                                phi_15312_ = f32();
                                                                phi_15313_ = true;
                                                            } else {
                                                                let _e5935_ = ((_e5918_ & 2147483648u) == 0u);
                                                                if _e5935_ {
                                                                    phi_15310_ = 0.999995f;
                                                                } else {
                                                                    phi_15310_ = f32();
                                                                }
                                                                let _e5937_ = phi_15310_;
                                                                phi_15312_ = _e5937_;
                                                                phi_15313_ = select(true, false, _e5935_);
                                                            }
                                                            let _e5940_ = phi_15312_;
                                                            let _e5942_ = phi_15313_;
                                                            if _e5942_ {
                                                                phi_15333_ = -0.999995f;
                                                            } else {
                                                                phi_15333_ = _e5940_;
                                                            }
                                                            let _e5944_ = phi_15333_;
                                                            phi_15392_ = _e5944_;
                                                        } else {
                                                            if (_e5917_ < -0.999995f) {
                                                                phi_15391_ = -0.999995f;
                                                            } else {
                                                                if (_e5917_ > 0.999995f) {
                                                                    phi_15390_ = 0.999995f;
                                                                } else {
                                                                    if _e5920_ {
                                                                        phi_15384_ = true;
                                                                    } else {
                                                                        phi_15384_ = (_e5919_ == 0u);
                                                                    }
                                                                    let _e5925_ = phi_15384_;
                                                                    phi_15390_ = bitcast<f32>(select(_e5918_, 0u, _e5925_));
                                                                }
                                                                let _e5929_ = phi_15390_;
                                                                phi_15391_ = _e5929_;
                                                            }
                                                            let _e5931_ = phi_15391_;
                                                            phi_15392_ = _e5931_;
                                                        }
                                                        let _e5946_ = phi_15392_;
                                                        phi_15395_ = type_28_(bitcast<f32>(select(_e5769_, 0u, _e5774_)), _e5946_);
                                                    } else {
                                                        let _e5618_ = local_44_;
                                                        phi_15395_ = type_28_(_e5618_, 1f);
                                                    }
                                                    let _e5949_ = phi_15395_;
                                                    if _e1589_ {
                                                        if ((bitcast<u32>(_e5949_.member) & 2139095040u) != 2139095040u) {
                                                            let _e5955_ = local_45_;
                                                            let _e5958_ = ((bitcast<u32>(_e5955_) & 2139095040u) != 2139095040u);
                                                            if _e5958_ {
                                                                let _e5959_ = abs(_e5949_.member);
                                                                let _e5961_ = local_46_;
                                                                let _e5962_ = abs(_e5961_);
                                                                let _e5964_ = select(_e5962_, _e5959_, (_e5959_ > _e5962_));
                                                                let _e5966_ = select(1f, _e5964_, (_e5964_ > 1f));
                                                                let _e5968_ = local_47_;
                                                                if ((bitcast<u32>(_e5966_) & 2139095040u) != 2139095040u) {
                                                                    let _e5974_ = (_e5966_ <= 0f);
                                                                    if _e5974_ {
                                                                        phi_15518_ = f32();
                                                                    } else {
                                                                        if (_e5966_ < 0.00001f) {
                                                                            let _e6006_ = (select(0.000000000000000000000001f, _e5966_, (_e5966_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                            if (_e6006_ != _e6006_) {
                                                                                phi_29176_ = true;
                                                                            } else {
                                                                                phi_29176_ = (0.000001f >= _e6006_);
                                                                            }
                                                                            let _e6010_ = phi_29176_;
                                                                            let _e6011_ = select(_e6006_, 0.000001f, _e6010_);
                                                                            let _e6014_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6011_)));
                                                                            let _e6018_ = (_e6014_ + (_e6014_ * (1f - (_e6011_ * _e6014_))));
                                                                            let _e6022_ = (_e6018_ + (_e6018_ * (1f - (_e6011_ * _e6018_))));
                                                                            let _e6028_ = bitcast<u32>(((_e6022_ + (_e6022_ * (1f - (_e6011_ * _e6022_)))) * 18446744000000000000f));
                                                                            let _e6029_ = (_e6028_ & 2139095040u);
                                                                            if (_e6029_ == 2139095040u) {
                                                                                phi_15476_ = true;
                                                                            } else {
                                                                                phi_15476_ = (_e6029_ == 0u);
                                                                            }
                                                                            let _e6033_ = phi_15476_;
                                                                            phi_15517_ = bitcast<f32>(select(_e6028_, 0u, _e6033_));
                                                                        } else {
                                                                            if (_e5966_ != _e5966_) {
                                                                                phi_29116_ = true;
                                                                            } else {
                                                                                phi_29116_ = (0.000001f >= _e5966_);
                                                                            }
                                                                            let _e5979_ = phi_29116_;
                                                                            let _e5980_ = select(_e5966_, 0.000001f, _e5979_);
                                                                            let _e5983_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5980_)));
                                                                            let _e5987_ = (_e5983_ + (_e5983_ * (1f - (_e5980_ * _e5983_))));
                                                                            let _e5991_ = (_e5987_ + (_e5987_ * (1f - (_e5980_ * _e5987_))));
                                                                            let _e5996_ = bitcast<u32>((_e5991_ + (_e5991_ * (1f - (_e5980_ * _e5991_)))));
                                                                            let _e5997_ = (_e5996_ & 2139095040u);
                                                                            if (_e5997_ == 2139095040u) {
                                                                                phi_15511_ = true;
                                                                            } else {
                                                                                phi_15511_ = (_e5997_ == 0u);
                                                                            }
                                                                            let _e6001_ = phi_15511_;
                                                                            phi_15517_ = bitcast<f32>(select(_e5996_, 0u, _e6001_));
                                                                        }
                                                                        let _e6037_ = phi_15517_;
                                                                        phi_15518_ = _e6037_;
                                                                    }
                                                                    let _e6039_ = phi_15518_;
                                                                    phi_15519_ = _e6039_;
                                                                    phi_15520_ = _e5974_;
                                                                } else {
                                                                    phi_15519_ = f32();
                                                                    phi_15520_ = true;
                                                                }
                                                                let _e6041_ = phi_15519_;
                                                                let _e6043_ = phi_15520_;
                                                                phi_15526_ = (abs((_e5949_.member - _e5968_)) * select(_e6041_, 0f, _e6043_));
                                                            } else {
                                                                phi_15526_ = f32();
                                                            }
                                                            let _e6047_ = phi_15526_;
                                                            phi_15528_ = _e6047_;
                                                            phi_15529_ = select(true, false, _e5958_);
                                                        } else {
                                                            phi_15528_ = f32();
                                                            phi_15529_ = true;
                                                        }
                                                        let _e6050_ = phi_15528_;
                                                        let _e6052_ = phi_15529_;
                                                        phi_15534_ = select(_e6050_, 1000000000f, _e6052_);
                                                    } else {
                                                        phi_15534_ = 0f;
                                                    }
                                                    let _e6055_ = phi_15534_;
                                                    let _e6056_ = (_e169_ < 128u);
                                                    if _e6056_ {
                                                        let _e6057_ = (32u + _e170_);
                                                        if (_e6057_ < 256u) {
                                                        } else {
                                                            phi_35659_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e6057_] = _e6055_;
                                                    }
                                                    workgroupBarrier();
                                                    if (_e169_ >= 128u) {
                                                        let _e6061_ = (32u + _e170_);
                                                        let _e6062_ = (_e6061_ < 256u);
                                                        if _e6062_ {
                                                        } else {
                                                            phi_35659_ = true;
                                                            break;
                                                        }
                                                        let _e6064_ = aux_scratch[_e6061_];
                                                        if _e6062_ {
                                                        } else {
                                                            phi_35659_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e6061_] = select(_e6064_, _e6055_, (_e6055_ > _e6064_));
                                                    }
                                                    workgroupBarrier();
                                                    phi_15572_ = 64u;
                                                    loop {
                                                        let _e6068_ = phi_15572_;
                                                        let _e6069_ = (_e6068_ > 0u);
                                                        if _e6069_ {
                                                            if (_e170_ < _e6068_) {
                                                                if _e6056_ {
                                                                    let _e6071_ = (32u + _e170_);
                                                                    let _e6072_ = (_e6071_ < 256u);
                                                                    if _e6072_ {
                                                                    } else {
                                                                        phi_35324_ = true;
                                                                        break;
                                                                    }
                                                                    let _e6074_ = aux_scratch[_e6071_];
                                                                    let _e6075_ = (_e6071_ + _e6068_);
                                                                    if (_e6075_ < 256u) {
                                                                    } else {
                                                                        phi_35324_ = true;
                                                                        break;
                                                                    }
                                                                    let _e6078_ = aux_scratch[_e6075_];
                                                                    if _e6072_ {
                                                                    } else {
                                                                        phi_35324_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e6071_] = select(_e6074_, _e6078_, (_e6078_ > _e6074_));
                                                                }
                                                            }
                                                            workgroupBarrier();
                                                            phi_15573_ = (_e6068_ >> bitcast<u32>(1i));
                                                        } else {
                                                            phi_15573_ = u32();
                                                        }
                                                        let _e6084_ = phi_15573_;
                                                        phi_15572_ = _e6084_;
                                                        phi_35324_ = _e5609_;
                                                        if !(_e6069_) {
                                                            break;
                                                        }
                                                    }
                                                    let _e6087_ = phi_35324_;
                                                    phi_35659_ = _e6087_;
                                                    if _e6087_ {
                                                        break;
                                                    }
                                                    let _e6089_ = aux_scratch[32];
                                                    workgroupBarrier();
                                                    if (_e6089_ > 0.0000005f) {
                                                        if (_e169_ < 2u) {
                                                            let _e6144_ = (_e169_ == 1u);
                                                            let _e6145_ = select(0u, 2u, _e6144_);
                                                            if _e6144_ {
                                                                phi_15652_ = (20u + _e4426_.member_1_);
                                                            } else {
                                                                phi_15652_ = (16u + _e4426_.member_1_);
                                                            }
                                                            let _e6150_ = phi_15652_;
                                                            let _e6151_ = (_e6150_ < 256u);
                                                            if _e6151_ {
                                                            } else {
                                                                phi_35659_ = true;
                                                                break;
                                                            }
                                                            let _e6153_ = aux_scratch[_e6150_];
                                                            phi_15662_ = type_15_(0u, _e1587_);
                                                            phi_15665_ = _e6153_;
                                                            loop {
                                                                let _e6156_ = phi_15662_;
                                                                let _e6158_ = phi_15665_;
                                                                local_48_ = _e6158_;
                                                                if (_e6156_.member < _e6156_.member_1_) {
                                                                    phi_15663_ = type_15_((_e6156_.member + 1u), _e6156_.member_1_);
                                                                    phi_15688_ = type_15_(1u, _e6156_.member);
                                                                } else {
                                                                    phi_15663_ = _e6156_;
                                                                    phi_15688_ = type_15_(0u, u32());
                                                                }
                                                                let _e6171_ = phi_15663_;
                                                                let _e6173_ = phi_15688_;
                                                                let _e6177_ = (bitcast<i32>(_e6173_.member) != 0i);
                                                                if _e6177_ {
                                                                    let _e6179_ = (_e4426_.member_1_ * 4u);
                                                                    let _e6182_ = ((_e6173_.member_1_ & 127u) & 127u);
                                                                    let _e6183_ = (((_e6179_ + _e6145_) * 128u) + _e6182_);
                                                                    if (_e6183_ < 2048u) {
                                                                    } else {
                                                                        phi_35559_ = true;
                                                                        break;
                                                                    }
                                                                    let _e6186_ = lane_buf[_e6183_];
                                                                    let _e6189_ = (((_e6179_ + (_e6145_ + 1u)) * 128u) + _e6182_);
                                                                    let _e6190_ = (_e6189_ < 2048u);
                                                                    if _e6190_ {
                                                                    } else {
                                                                        phi_35559_ = true;
                                                                        break;
                                                                    }
                                                                    let _e6192_ = lane_buf[_e6189_];
                                                                    let _e6195_ = bitcast<u32>((_e4568_ * _e6192_));
                                                                    let _e6196_ = (_e6195_ & 2139095040u);
                                                                    if (_e6196_ == 2139095040u) {
                                                                        phi_15733_ = true;
                                                                    } else {
                                                                        phi_15733_ = (_e6196_ == 0u);
                                                                    }
                                                                    let _e6200_ = phi_15733_;
                                                                    let _e6210_ = ((1f - _e4504_) + select((_e4542_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e6195_, 0u, _e6200_))) & 2147483647u))), 0f, (_e4542_ <= 0f)));
                                                                    if (_e6210_ != _e6210_) {
                                                                        phi_29236_ = true;
                                                                    } else {
                                                                        phi_29236_ = (0f >= _e6210_);
                                                                    }
                                                                    let _e6214_ = phi_29236_;
                                                                    let _e6215_ = select(_e6210_, 0f, _e6214_);
                                                                    if (_e6215_ != _e6215_) {
                                                                        phi_29296_ = true;
                                                                    } else {
                                                                        phi_29296_ = (1f <= _e6215_);
                                                                    }
                                                                    let _e6219_ = phi_29296_;
                                                                    let _e6220_ = select(_e6215_, 1f, _e6219_);
                                                                    let _e6221_ = (_e4485_ * _e6158_);
                                                                    let _e6222_ = (1f - _e4561_);
                                                                    let _e6225_ = ((_e4575_ * _e6186_) - ((_e4523_ * _e6222_) * _e6221_));
                                                                    if (_e6225_ != _e6225_) {
                                                                        phi_29356_ = true;
                                                                    } else {
                                                                        phi_29356_ = (-1f >= _e6225_);
                                                                    }
                                                                    let _e6229_ = phi_29356_;
                                                                    let _e6230_ = select(_e6225_, -1f, _e6229_);
                                                                    if (_e6230_ != _e6230_) {
                                                                        phi_29416_ = true;
                                                                    } else {
                                                                        phi_29416_ = (1f <= _e6230_);
                                                                    }
                                                                    let _e6234_ = phi_29416_;
                                                                    let _e6235_ = select(_e6230_, 1f, _e6234_);
                                                                    let _e6242_ = bitcast<f32>((bitcast<u32>(_e6225_) & 2147483647u));
                                                                    if (_e6242_ != _e6242_) {
                                                                        phi_29476_ = true;
                                                                    } else {
                                                                        phi_29476_ = (1000000000000000f <= _e6242_);
                                                                    }
                                                                    let _e6246_ = phi_29476_;
                                                                    let _e6247_ = select(_e6242_, 1000000000000000f, _e6246_);
                                                                    let _e6248_ = (_e6247_ * _e6247_);
                                                                    let _e6249_ = (_e6248_ + 0.0000010000001f);
                                                                    if (_e6249_ != _e6249_) {
                                                                        phi_29536_ = true;
                                                                    } else {
                                                                        phi_29536_ = (0.000000000001f >= _e6249_);
                                                                    }
                                                                    let _e6253_ = phi_29536_;
                                                                    let _e6254_ = select(_e6249_, 0.000000000001f, _e6253_);
                                                                    let _e6255_ = (_e6254_ * 0.5f);
                                                                    let _e6260_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e6254_) >> bitcast<u32>(1i))));
                                                                    let _e6264_ = (_e6260_ * (1.5f - (_e6255_ * (_e6260_ * _e6260_))));
                                                                    let _e6268_ = (_e6264_ * (1.5f - (_e6255_ * (_e6264_ * _e6264_))));
                                                                    let _e6273_ = (_e6249_ * (_e6268_ * (1.5f - (_e6255_ * (_e6268_ * _e6268_)))));
                                                                    let _e6274_ = (_e6273_ + 0.001f);
                                                                    if (_e6274_ != _e6274_) {
                                                                        phi_29596_ = true;
                                                                    } else {
                                                                        phi_29596_ = (0.000001f >= _e6274_);
                                                                    }
                                                                    let _e6278_ = phi_29596_;
                                                                    let _e6279_ = select(_e6274_, 0.000001f, _e6278_);
                                                                    let _e6282_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6279_)));
                                                                    let _e6286_ = (_e6282_ + (_e6282_ * (1f - (_e6279_ * _e6282_))));
                                                                    let _e6290_ = (_e6286_ + (_e6286_ * (1f - (_e6279_ * _e6286_))));
                                                                    let _e6294_ = (_e6290_ + (_e6290_ * (1f - (_e6279_ * _e6290_))));
                                                                    let _e6295_ = (_e6273_ + _e6247_);
                                                                    if (_e6295_ != _e6295_) {
                                                                        phi_29656_ = true;
                                                                    } else {
                                                                        phi_29656_ = (0.000001f >= _e6295_);
                                                                    }
                                                                    let _e6299_ = phi_29656_;
                                                                    let _e6300_ = select(_e6295_, 0.000001f, _e6299_);
                                                                    let _e6303_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6300_)));
                                                                    let _e6307_ = (_e6303_ + (_e6303_ * (1f - (_e6300_ * _e6303_))));
                                                                    let _e6311_ = (_e6307_ + (_e6307_ * (1f - (_e6300_ * _e6307_))));
                                                                    let _e6321_ = (_e6242_ <= 1000000000000000f);
                                                                    let _e6331_ = select(select(select(_e6242_, (0.5f * (_e6247_ + (_e6248_ * _e6294_))), _e6321_), select(-0.0005f, -(((0.0005f * (_e6247_ * (_e6311_ + (_e6311_ * (1f - (_e6300_ * _e6311_)))))) * (1f + (_e6247_ * _e6294_)))), _e6321_), (_e6225_ <= 0f)), 0f, (_e6242_ <= 0f));
                                                                    let _e6342_ = bitcast<u32>((((1f - (_e4561_ * _e6220_)) * _e6221_) + (_e6220_ * select(((_e6222_ * _e6331_) + (_e4561_ * (_e6235_ * (1f - (0.33333334f * (_e6235_ * _e6235_)))))), _e6331_, (_e4561_ <= 0f)))));
                                                                    let _e6343_ = (_e6342_ & 2139095040u);
                                                                    if (_e6343_ == 2139095040u) {
                                                                        phi_16483_ = true;
                                                                    } else {
                                                                        phi_16483_ = (_e6343_ == 0u);
                                                                    }
                                                                    let _e6347_ = phi_16483_;
                                                                    let _e6349_ = bitcast<f32>(select(_e6342_, 0u, _e6347_));
                                                                    if _e6190_ {
                                                                    } else {
                                                                        phi_35559_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e6189_] = _e6349_;
                                                                    phi_15666_ = _e6349_;
                                                                } else {
                                                                    phi_15666_ = f32();
                                                                }
                                                                let _e6351_ = phi_15666_;
                                                                phi_15662_ = _e6171_;
                                                                phi_15665_ = _e6351_;
                                                                phi_35559_ = _e6087_;
                                                                if !(_e6177_) {
                                                                    break;
                                                                }
                                                            }
                                                            let _e6354_ = phi_35559_;
                                                            phi_35659_ = _e6354_;
                                                            if _e6354_ {
                                                                break;
                                                            }
                                                            let _e6356_ = local_48_;
                                                            let _e6357_ = bitcast<u32>(_e6356_);
                                                            let _e6358_ = (_e6357_ & 2139095040u);
                                                            if (_e6358_ == 2139095040u) {
                                                                phi_16541_ = true;
                                                            } else {
                                                                phi_16541_ = (_e6358_ == 0u);
                                                            }
                                                            let _e6362_ = phi_16541_;
                                                            if _e6151_ {
                                                            } else {
                                                                phi_35659_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e6150_] = bitcast<f32>(select(_e6357_, 0u, _e6362_));
                                                            phi_35671_ = _e6354_;
                                                        } else {
                                                            phi_35671_ = _e6087_;
                                                        }
                                                        let _e6366_ = phi_35671_;
                                                        workgroupBarrier();
                                                        phi_35662_ = _e6366_;
                                                    } else {
                                                        workgroupBarrier();
                                                        if _e6056_ {
                                                            if _e1589_ {
                                                                let _e6091_ = (512u + _e170_);
                                                                if (_e6091_ < 1536u) {
                                                                } else {
                                                                    phi_35659_ = true;
                                                                    break;
                                                                }
                                                                let _e6094_ = scan_scratch[_e6091_];
                                                                let _e6095_ = (_e4426_.member_1_ * 4u);
                                                                let _e6098_ = (_e170_ & 127u);
                                                                let _e6099_ = (((_e6095_ + 1u) * 128u) + _e6098_);
                                                                if (_e6099_ < 2048u) {
                                                                } else {
                                                                    phi_35659_ = true;
                                                                    break;
                                                                }
                                                                lane_buf[_e6099_] = _e6094_;
                                                                let _e6102_ = (640u + _e170_);
                                                                if (_e6102_ < 1536u) {
                                                                } else {
                                                                    phi_35659_ = true;
                                                                    break;
                                                                }
                                                                let _e6105_ = scan_scratch[_e6102_];
                                                                let _e6108_ = (((_e6095_ + 3u) * 128u) + _e6098_);
                                                                if (_e6108_ < 2048u) {
                                                                } else {
                                                                    phi_35659_ = true;
                                                                    break;
                                                                }
                                                                lane_buf[_e6108_] = _e6105_;
                                                            }
                                                        }
                                                        if _e1190_ {
                                                            if _e1588_ {
                                                                let _e6112_ = ((_e1587_ - 1u) & 127u);
                                                                let _e6113_ = (512u + _e6112_);
                                                                if (_e6113_ < 1536u) {
                                                                } else {
                                                                    phi_35659_ = true;
                                                                    break;
                                                                }
                                                                let _e6116_ = scan_scratch[_e6113_];
                                                                let _e6117_ = bitcast<u32>(_e6116_);
                                                                let _e6118_ = (_e6117_ & 2139095040u);
                                                                if (_e6118_ == 2139095040u) {
                                                                    phi_16626_ = true;
                                                                } else {
                                                                    phi_16626_ = (_e6118_ == 0u);
                                                                }
                                                                let _e6122_ = phi_16626_;
                                                                let _e6125_ = (16u + _e4426_.member_1_);
                                                                if (_e6125_ < 256u) {
                                                                } else {
                                                                    phi_35659_ = true;
                                                                    break;
                                                                }
                                                                aux_scratch[_e6125_] = bitcast<f32>(select(_e6117_, 0u, _e6122_));
                                                                let _e6128_ = (640u + _e6112_);
                                                                if (_e6128_ < 1536u) {
                                                                } else {
                                                                    phi_35659_ = true;
                                                                    break;
                                                                }
                                                                let _e6131_ = scan_scratch[_e6128_];
                                                                let _e6132_ = bitcast<u32>(_e6131_);
                                                                let _e6133_ = (_e6132_ & 2139095040u);
                                                                if (_e6133_ == 2139095040u) {
                                                                    phi_16657_ = true;
                                                                } else {
                                                                    phi_16657_ = (_e6133_ == 0u);
                                                                }
                                                                let _e6137_ = phi_16657_;
                                                                let _e6140_ = (20u + _e4426_.member_1_);
                                                                if (_e6140_ < 256u) {
                                                                } else {
                                                                    phi_35659_ = true;
                                                                    break;
                                                                }
                                                                aux_scratch[_e6140_] = bitcast<f32>(select(_e6132_, 0u, _e6137_));
                                                            }
                                                        }
                                                        workgroupBarrier();
                                                        phi_35662_ = _e6087_;
                                                    }
                                                    let _e6368_ = phi_35662_;
                                                    phi_35661_ = _e6368_;
                                                }
                                                let _e6370_ = phi_35661_;
                                                phi_35660_ = _e6370_;
                                            } else {
                                                phi_35660_ = _e4409_;
                                            }
                                            let _e6372_ = phi_35660_;
                                            phi_34906_ = _e6372_;
                                            phi_9272_ = _e4424_;
                                            phi_35659_ = _e6372_;
                                            if !(_e4430_) {
                                                break;
                                            }
                                        }
                                        let _e6375_ = phi_35659_;
                                        phi_36576_ = _e6375_;
                                        if _e6375_ {
                                            break;
                                        }
                                        phi_35678_ = _e6375_;
                                    } else {
                                        if (_e169_ < 8u) {
                                            let _e4032_ = (_e169_ >> bitcast<u32>(1i));
                                            let _e4034_ = ((_e169_ & 1u) != 0u);
                                            let _e4035_ = (_e4032_ & 3u);
                                            switch bitcast<i32>(_e4035_) {
                                                case 0: {
                                                    phi_16687_ = _e382_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16687_ = _e382_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16687_ = _e382_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16687_ = _e382_.w;
                                                    break;
                                                }
                                            }
                                            let _e4042_ = phi_16687_;
                                            switch bitcast<i32>(_e4035_) {
                                                case 0: {
                                                    phi_16698_ = _e389_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16698_ = _e389_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16698_ = _e389_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16698_ = _e389_.w;
                                                    break;
                                                }
                                            }
                                            let _e4049_ = phi_16698_;
                                            switch bitcast<i32>(_e4035_) {
                                                case 0: {
                                                    phi_16709_ = _e396_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16709_ = _e396_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16709_ = _e396_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16709_ = _e396_.w;
                                                    break;
                                                }
                                            }
                                            let _e4056_ = phi_16709_;
                                            switch bitcast<i32>(_e4035_) {
                                                case 0: {
                                                    phi_16720_ = _e403_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16720_ = _e403_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16720_ = _e403_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16720_ = _e403_.w;
                                                    break;
                                                }
                                            }
                                            let _e4063_ = phi_16720_;
                                            switch bitcast<i32>(_e4035_) {
                                                case 0: {
                                                    phi_16731_ = _e410_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16731_ = _e410_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16731_ = _e410_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16731_ = _e410_.w;
                                                    break;
                                                }
                                            }
                                            let _e4070_ = phi_16731_;
                                            let _e4071_ = bitcast<u32>(_e4042_);
                                            let _e4072_ = (_e4071_ & 2139095040u);
                                            if (_e4072_ != 2139095040u) {
                                                if (_e4042_ <= 0f) {
                                                    phi_16845_ = 0f;
                                                } else {
                                                    if (_e4042_ >= 0.999995f) {
                                                        phi_16844_ = 0.999995f;
                                                    } else {
                                                        if (_e4072_ == 2139095040u) {
                                                            phi_16838_ = true;
                                                        } else {
                                                            phi_16838_ = (_e4072_ == 0u);
                                                        }
                                                        let _e4081_ = phi_16838_;
                                                        phi_16844_ = bitcast<f32>(select(_e4071_, 0u, _e4081_));
                                                    }
                                                    let _e4085_ = phi_16844_;
                                                    phi_16845_ = _e4085_;
                                                }
                                                let _e4087_ = phi_16845_;
                                                phi_16852_ = _e4087_;
                                            } else {
                                                phi_16852_ = select(0f, 0.999995f, (_e4071_ == 2139095040u));
                                            }
                                            let _e4089_ = phi_16852_;
                                            let _e4090_ = bitcast<u32>(_e4049_);
                                            let _e4091_ = (_e4090_ & 2139095040u);
                                            if (_e4091_ != 2139095040u) {
                                                if (_e4049_ <= 0f) {
                                                    phi_16884_ = 0f;
                                                } else {
                                                    if (_e4049_ >= 1f) {
                                                        phi_16883_ = 1f;
                                                    } else {
                                                        if (_e4091_ == 2139095040u) {
                                                            phi_16877_ = true;
                                                        } else {
                                                            phi_16877_ = (_e4091_ == 0u);
                                                        }
                                                        let _e4100_ = phi_16877_;
                                                        phi_16883_ = bitcast<f32>(select(_e4090_, 0u, _e4100_));
                                                    }
                                                    let _e4104_ = phi_16883_;
                                                    phi_16884_ = _e4104_;
                                                }
                                                let _e4106_ = phi_16884_;
                                                phi_16891_ = _e4106_;
                                            } else {
                                                phi_16891_ = select(0f, 1f, (_e4090_ == 2139095040u));
                                            }
                                            let _e4108_ = phi_16891_;
                                            let _e4109_ = bitcast<u32>(_e4056_);
                                            let _e4110_ = (_e4109_ & 2139095040u);
                                            if (_e4110_ != 2139095040u) {
                                                if (_e4056_ <= 0f) {
                                                    phi_16923_ = 0f;
                                                } else {
                                                    if (_e4056_ >= 1f) {
                                                        phi_16922_ = 1f;
                                                    } else {
                                                        if (_e4110_ == 2139095040u) {
                                                            phi_16916_ = true;
                                                        } else {
                                                            phi_16916_ = (_e4110_ == 0u);
                                                        }
                                                        let _e4119_ = phi_16916_;
                                                        phi_16922_ = bitcast<f32>(select(_e4109_, 0u, _e4119_));
                                                    }
                                                    let _e4123_ = phi_16922_;
                                                    phi_16923_ = _e4123_;
                                                }
                                                let _e4125_ = phi_16923_;
                                                phi_16930_ = _e4125_;
                                            } else {
                                                phi_16930_ = select(0f, 1f, (_e4109_ == 2139095040u));
                                            }
                                            let _e4127_ = phi_16930_;
                                            let _e4128_ = bitcast<u32>(_e4063_);
                                            let _e4129_ = (_e4128_ & 2139095040u);
                                            if (_e4129_ != 2139095040u) {
                                                if (_e4063_ <= 0f) {
                                                    phi_16962_ = 0f;
                                                } else {
                                                    if (_e4063_ >= 64f) {
                                                        phi_16961_ = 64f;
                                                    } else {
                                                        if (_e4129_ == 2139095040u) {
                                                            phi_16955_ = true;
                                                        } else {
                                                            phi_16955_ = (_e4129_ == 0u);
                                                        }
                                                        let _e4138_ = phi_16955_;
                                                        phi_16961_ = bitcast<f32>(select(_e4128_, 0u, _e4138_));
                                                    }
                                                    let _e4142_ = phi_16961_;
                                                    phi_16962_ = _e4142_;
                                                }
                                                let _e4144_ = phi_16962_;
                                                phi_16969_ = _e4144_;
                                            } else {
                                                phi_16969_ = select(0f, 64f, (_e4128_ == 2139095040u));
                                            }
                                            let _e4146_ = phi_16969_;
                                            let _e4147_ = bitcast<u32>(_e4070_);
                                            let _e4148_ = (_e4147_ & 2139095040u);
                                            if (_e4148_ != 2139095040u) {
                                                if (_e4070_ <= 0f) {
                                                    phi_17001_ = 0f;
                                                } else {
                                                    if (_e4070_ >= 1f) {
                                                        phi_17000_ = 1f;
                                                    } else {
                                                        if (_e4148_ == 2139095040u) {
                                                            phi_16994_ = true;
                                                        } else {
                                                            phi_16994_ = (_e4148_ == 0u);
                                                        }
                                                        let _e4157_ = phi_16994_;
                                                        phi_17000_ = bitcast<f32>(select(_e4147_, 0u, _e4157_));
                                                    }
                                                    let _e4161_ = phi_17000_;
                                                    phi_17001_ = _e4161_;
                                                }
                                                let _e4163_ = phi_17001_;
                                                phi_17008_ = _e4163_;
                                            } else {
                                                phi_17008_ = select(0f, 1f, (_e4147_ == 2139095040u));
                                            }
                                            let _e4165_ = phi_17008_;
                                            switch bitcast<i32>(_e4035_) {
                                                case 0: {
                                                    phi_17118_ = _e424_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_17118_ = _e424_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_17118_ = _e424_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_17118_ = _e424_.w;
                                                    break;
                                                }
                                            }
                                            let _e4172_ = phi_17118_;
                                            switch bitcast<i32>(_e4035_) {
                                                case 0: {
                                                    phi_17129_ = _e431_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_17129_ = _e431_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_17129_ = _e431_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_17129_ = _e431_.w;
                                                    break;
                                                }
                                            }
                                            let _e4179_ = phi_17129_;
                                            if (_e4127_ == 0f) {
                                                phi_17199_ = true;
                                            } else {
                                                phi_17199_ = (_e4165_ >= 1f);
                                            }
                                            let _e4183_ = phi_17199_;
                                            if _e4183_ {
                                                phi_35680_ = _e4029_;
                                            } else {
                                                let _e4184_ = select(0u, 2u, _e4034_);
                                                if _e4034_ {
                                                    phi_17225_ = (20u + _e4032_);
                                                } else {
                                                    phi_17225_ = (16u + _e4032_);
                                                }
                                                let _e4189_ = phi_17225_;
                                                let _e4190_ = (_e4189_ < 256u);
                                                if _e4190_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                let _e4192_ = aux_scratch[_e4189_];
                                                phi_17235_ = type_15_(0u, _e1587_);
                                                phi_17238_ = _e4192_;
                                                loop {
                                                    let _e4195_ = phi_17235_;
                                                    let _e4197_ = phi_17238_;
                                                    local_41_ = _e4197_;
                                                    if (_e4195_.member < _e4195_.member_1_) {
                                                        phi_17236_ = type_15_((_e4195_.member + 1u), _e4195_.member_1_);
                                                        phi_17261_ = type_15_(1u, _e4195_.member);
                                                    } else {
                                                        phi_17236_ = _e4195_;
                                                        phi_17261_ = type_15_(0u, u32());
                                                    }
                                                    let _e4210_ = phi_17236_;
                                                    let _e4212_ = phi_17261_;
                                                    let _e4216_ = (bitcast<i32>(_e4212_.member) != 0i);
                                                    if _e4216_ {
                                                        let _e4218_ = (_e4032_ * 4u);
                                                        let _e4221_ = ((_e4212_.member_1_ & 127u) & 127u);
                                                        let _e4222_ = (((_e4218_ + _e4184_) * 128u) + _e4221_);
                                                        if (_e4222_ < 2048u) {
                                                        } else {
                                                            phi_34724_ = true;
                                                            break;
                                                        }
                                                        let _e4225_ = lane_buf[_e4222_];
                                                        let _e4228_ = (((_e4218_ + (_e4184_ + 1u)) * 128u) + _e4221_);
                                                        let _e4229_ = (_e4228_ < 2048u);
                                                        if _e4229_ {
                                                        } else {
                                                            phi_34724_ = true;
                                                            break;
                                                        }
                                                        let _e4231_ = lane_buf[_e4228_];
                                                        let _e4234_ = bitcast<u32>((_e4172_ * _e4231_));
                                                        let _e4235_ = (_e4234_ & 2139095040u);
                                                        if (_e4235_ == 2139095040u) {
                                                            phi_17306_ = true;
                                                        } else {
                                                            phi_17306_ = (_e4235_ == 0u);
                                                        }
                                                        let _e4239_ = phi_17306_;
                                                        let _e4249_ = ((1f - _e4108_) + select((_e4146_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e4234_, 0u, _e4239_))) & 2147483647u))), 0f, (_e4146_ <= 0f)));
                                                        if (_e4249_ != _e4249_) {
                                                            phi_25396_ = true;
                                                        } else {
                                                            phi_25396_ = (0f >= _e4249_);
                                                        }
                                                        let _e4253_ = phi_25396_;
                                                        let _e4254_ = select(_e4249_, 0f, _e4253_);
                                                        if (_e4254_ != _e4254_) {
                                                            phi_25456_ = true;
                                                        } else {
                                                            phi_25456_ = (1f <= _e4254_);
                                                        }
                                                        let _e4258_ = phi_25456_;
                                                        let _e4259_ = select(_e4254_, 1f, _e4258_);
                                                        let _e4260_ = (_e4089_ * _e4197_);
                                                        let _e4261_ = (1f - _e4165_);
                                                        let _e4264_ = ((_e4179_ * _e4225_) - ((_e4127_ * _e4261_) * _e4260_));
                                                        if (_e4264_ != _e4264_) {
                                                            phi_25516_ = true;
                                                        } else {
                                                            phi_25516_ = (-1f >= _e4264_);
                                                        }
                                                        let _e4268_ = phi_25516_;
                                                        let _e4269_ = select(_e4264_, -1f, _e4268_);
                                                        if (_e4269_ != _e4269_) {
                                                            phi_25576_ = true;
                                                        } else {
                                                            phi_25576_ = (1f <= _e4269_);
                                                        }
                                                        let _e4273_ = phi_25576_;
                                                        let _e4274_ = select(_e4269_, 1f, _e4273_);
                                                        let _e4281_ = bitcast<f32>((bitcast<u32>(_e4264_) & 2147483647u));
                                                        if (_e4281_ != _e4281_) {
                                                            phi_25636_ = true;
                                                        } else {
                                                            phi_25636_ = (1000000000000000f <= _e4281_);
                                                        }
                                                        let _e4285_ = phi_25636_;
                                                        let _e4286_ = select(_e4281_, 1000000000000000f, _e4285_);
                                                        let _e4287_ = (_e4286_ * _e4286_);
                                                        let _e4288_ = (_e4287_ + 0.0000010000001f);
                                                        if (_e4288_ != _e4288_) {
                                                            phi_25696_ = true;
                                                        } else {
                                                            phi_25696_ = (0.000000000001f >= _e4288_);
                                                        }
                                                        let _e4292_ = phi_25696_;
                                                        let _e4293_ = select(_e4288_, 0.000000000001f, _e4292_);
                                                        let _e4294_ = (_e4293_ * 0.5f);
                                                        let _e4299_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e4293_) >> bitcast<u32>(1i))));
                                                        let _e4303_ = (_e4299_ * (1.5f - (_e4294_ * (_e4299_ * _e4299_))));
                                                        let _e4307_ = (_e4303_ * (1.5f - (_e4294_ * (_e4303_ * _e4303_))));
                                                        let _e4312_ = (_e4288_ * (_e4307_ * (1.5f - (_e4294_ * (_e4307_ * _e4307_)))));
                                                        let _e4313_ = (_e4312_ + 0.001f);
                                                        if (_e4313_ != _e4313_) {
                                                            phi_25756_ = true;
                                                        } else {
                                                            phi_25756_ = (0.000001f >= _e4313_);
                                                        }
                                                        let _e4317_ = phi_25756_;
                                                        let _e4318_ = select(_e4313_, 0.000001f, _e4317_);
                                                        let _e4321_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4318_)));
                                                        let _e4325_ = (_e4321_ + (_e4321_ * (1f - (_e4318_ * _e4321_))));
                                                        let _e4329_ = (_e4325_ + (_e4325_ * (1f - (_e4318_ * _e4325_))));
                                                        let _e4333_ = (_e4329_ + (_e4329_ * (1f - (_e4318_ * _e4329_))));
                                                        let _e4334_ = (_e4312_ + _e4286_);
                                                        if (_e4334_ != _e4334_) {
                                                            phi_25816_ = true;
                                                        } else {
                                                            phi_25816_ = (0.000001f >= _e4334_);
                                                        }
                                                        let _e4338_ = phi_25816_;
                                                        let _e4339_ = select(_e4334_, 0.000001f, _e4338_);
                                                        let _e4342_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4339_)));
                                                        let _e4346_ = (_e4342_ + (_e4342_ * (1f - (_e4339_ * _e4342_))));
                                                        let _e4350_ = (_e4346_ + (_e4346_ * (1f - (_e4339_ * _e4346_))));
                                                        let _e4360_ = (_e4281_ <= 1000000000000000f);
                                                        let _e4370_ = select(select(select(_e4281_, (0.5f * (_e4286_ + (_e4287_ * _e4333_))), _e4360_), select(-0.0005f, -(((0.0005f * (_e4286_ * (_e4350_ + (_e4350_ * (1f - (_e4339_ * _e4350_)))))) * (1f + (_e4286_ * _e4333_)))), _e4360_), (_e4264_ <= 0f)), 0f, (_e4281_ <= 0f));
                                                        let _e4381_ = bitcast<u32>((((1f - (_e4165_ * _e4259_)) * _e4260_) + (_e4259_ * select(((_e4261_ * _e4370_) + (_e4165_ * (_e4274_ * (1f - (0.33333334f * (_e4274_ * _e4274_)))))), _e4370_, (_e4165_ <= 0f)))));
                                                        let _e4382_ = (_e4381_ & 2139095040u);
                                                        if (_e4382_ == 2139095040u) {
                                                            phi_18056_ = true;
                                                        } else {
                                                            phi_18056_ = (_e4382_ == 0u);
                                                        }
                                                        let _e4386_ = phi_18056_;
                                                        let _e4388_ = bitcast<f32>(select(_e4381_, 0u, _e4386_));
                                                        if _e4229_ {
                                                        } else {
                                                            phi_34724_ = true;
                                                            break;
                                                        }
                                                        lane_buf[_e4228_] = _e4388_;
                                                        phi_17239_ = _e4388_;
                                                    } else {
                                                        phi_17239_ = f32();
                                                    }
                                                    let _e4390_ = phi_17239_;
                                                    phi_17235_ = _e4210_;
                                                    phi_17238_ = _e4390_;
                                                    phi_34724_ = _e4029_;
                                                    if !(_e4216_) {
                                                        break;
                                                    }
                                                }
                                                let _e4393_ = phi_34724_;
                                                phi_36576_ = _e4393_;
                                                if _e4393_ {
                                                    break;
                                                }
                                                let _e4395_ = local_41_;
                                                let _e4396_ = bitcast<u32>(_e4395_);
                                                let _e4397_ = (_e4396_ & 2139095040u);
                                                if (_e4397_ == 2139095040u) {
                                                    phi_18114_ = true;
                                                } else {
                                                    phi_18114_ = (_e4397_ == 0u);
                                                }
                                                let _e4401_ = phi_18114_;
                                                if _e4190_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                aux_scratch[_e4189_] = bitcast<f32>(select(_e4396_, 0u, _e4401_));
                                                phi_35680_ = _e4393_;
                                            }
                                            let _e4405_ = phi_35680_;
                                            phi_35679_ = _e4405_;
                                        } else {
                                            phi_35679_ = _e4029_;
                                        }
                                        let _e4407_ = phi_35679_;
                                        workgroupBarrier();
                                        phi_35678_ = _e4407_;
                                    }
                                    let _e6377_ = phi_35678_;
                                    local_9_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    local_10_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    local_11_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    local_12_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                    let _e6378_ = (_e169_ < 128u);
                                    if _e6378_ {
                                        if _e1589_ {
                                            local_13_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_14_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_15_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_16_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            phi_18135_ = type_15_(0u, 4u);
                                            loop {
                                                let _e6380_ = phi_18135_;
                                                if (_e6380_.member < _e6380_.member_1_) {
                                                    phi_18136_ = type_15_((_e6380_.member + 1u), _e6380_.member_1_);
                                                    phi_18159_ = type_15_(1u, _e6380_.member);
                                                } else {
                                                    phi_18136_ = _e6380_;
                                                    phi_18159_ = type_15_(0u, u32());
                                                }
                                                let _e6393_ = phi_18136_;
                                                let _e6395_ = phi_18159_;
                                                let _e6399_ = (bitcast<i32>(_e6395_.member) != 0i);
                                                if _e6399_ {
                                                    let _e6400_ = (_e6395_.member_1_ * 4u);
                                                    let _e6402_ = (_e170_ & 127u);
                                                    let _e6403_ = ((_e6395_.member_1_ * 512u) + _e6402_);
                                                    if (_e6403_ < 2048u) {
                                                    } else {
                                                        phi_35674_ = true;
                                                        break;
                                                    }
                                                    let _e6406_ = lane_buf[_e6403_];
                                                    let _e6407_ = (_e6395_.member_1_ < 4u);
                                                    if _e6407_ {
                                                    } else {
                                                        phi_35674_ = true;
                                                        break;
                                                    }
                                                    local_13_[_e6395_.member_1_] = _e6406_;
                                                    let _e6411_ = (((_e6400_ + 1u) * 128u) + _e6402_);
                                                    if (_e6411_ < 2048u) {
                                                    } else {
                                                        phi_35674_ = true;
                                                        break;
                                                    }
                                                    let _e6414_ = lane_buf[_e6411_];
                                                    if _e6407_ {
                                                    } else {
                                                        phi_35674_ = true;
                                                        break;
                                                    }
                                                    local_15_[_e6395_.member_1_] = _e6414_;
                                                    let _e6418_ = (((_e6400_ + 2u) * 128u) + _e6402_);
                                                    if (_e6418_ < 2048u) {
                                                    } else {
                                                        phi_35674_ = true;
                                                        break;
                                                    }
                                                    let _e6421_ = lane_buf[_e6418_];
                                                    if _e6407_ {
                                                    } else {
                                                        phi_35674_ = true;
                                                        break;
                                                    }
                                                    local_14_[_e6395_.member_1_] = _e6421_;
                                                    let _e6425_ = (((_e6400_ + 3u) * 128u) + _e6402_);
                                                    if (_e6425_ < 2048u) {
                                                    } else {
                                                        phi_35674_ = true;
                                                        break;
                                                    }
                                                    let _e6428_ = lane_buf[_e6425_];
                                                    if _e6407_ {
                                                    } else {
                                                        phi_35674_ = true;
                                                        break;
                                                    }
                                                    local_16_[_e6395_.member_1_] = _e6428_;
                                                }
                                                phi_18135_ = _e6393_;
                                                phi_35674_ = _e6377_;
                                                if !(_e6399_) {
                                                    break;
                                                }
                                            }
                                            let _e6432_ = phi_35674_;
                                            phi_36576_ = _e6432_;
                                            if _e6432_ {
                                                break;
                                            }
                                            let _e6433_ = local_15_;
                                            let _e6438_ = local_16_;
                                            let _e6447_ = (0.5f * (_e6433_[0] + _e6438_[0]));
                                            let _e6448_ = (0.5f * (_e6433_[1] + _e6438_[1]));
                                            let _e6449_ = (0.5f * (_e6433_[2] + _e6438_[2]));
                                            let _e6450_ = (0.5f * (_e6433_[3] + _e6438_[3]));
                                            local_17_ = array<f32, 4>((_e6433_[0] + (_e575_ * (_e6447_ - _e6433_[0]))), (_e6433_[1] + (_e595_ * (_e6448_ - _e6433_[1]))), (_e6433_[2] + (_e615_ * (_e6449_ - _e6433_[2]))), (_e6433_[3] + (_e635_ * (_e6450_ - _e6433_[3]))));
                                            local_18_ = array<f32, 4>((_e6438_[0] + (_e575_ * (_e6447_ - _e6438_[0]))), (_e6438_[1] + (_e595_ * (_e6448_ - _e6438_[1]))), (_e6438_[2] + (_e615_ * (_e6449_ - _e6438_[2]))), (_e6438_[3] + (_e635_ * (_e6450_ - _e6438_[3]))));
                                            phi_18331_ = type_15_(0u, 4u);
                                            loop {
                                                let _e6478_ = phi_18331_;
                                                if (_e6478_.member < _e6478_.member_1_) {
                                                    phi_18332_ = type_15_((_e6478_.member + 1u), _e6478_.member_1_);
                                                    phi_18355_ = type_15_(1u, _e6478_.member);
                                                } else {
                                                    phi_18332_ = _e6478_;
                                                    phi_18355_ = type_15_(0u, u32());
                                                }
                                                let _e6491_ = phi_18332_;
                                                let _e6493_ = phi_18355_;
                                                let _e6497_ = (bitcast<i32>(_e6493_.member) != 0i);
                                                if _e6497_ {
                                                    let _e6498_ = (_e6493_.member_1_ & 3u);
                                                    switch bitcast<i32>(_e6498_) {
                                                        case 0: {
                                                            phi_18726_ = _e438_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_18726_ = _e438_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_18726_ = _e438_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_18726_ = _e438_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e6505_ = phi_18726_;
                                                    switch bitcast<i32>(_e6498_) {
                                                        case 0: {
                                                            phi_18737_ = _e445_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_18737_ = _e445_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_18737_ = _e445_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_18737_ = _e445_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e6512_ = phi_18737_;
                                                    switch bitcast<i32>(_e6498_) {
                                                        case 0: {
                                                            phi_18748_ = _e452_.x;
                                                            break;
                                                        }
                                                        case 1: {
                                                            phi_18748_ = _e452_.y;
                                                            break;
                                                        }
                                                        case 2: {
                                                            phi_18748_ = _e452_.z;
                                                            break;
                                                        }
                                                        default: {
                                                            phi_18748_ = _e452_.w;
                                                            break;
                                                        }
                                                    }
                                                    let _e6519_ = phi_18748_;
                                                    let _e6520_ = (_e6493_.member_1_ < 4u);
                                                    if _e6520_ {
                                                    } else {
                                                        phi_35683_ = true;
                                                        break;
                                                    }
                                                    let _e6522_ = local_13_[_e6493_.member_1_];
                                                    if _e6520_ {
                                                    } else {
                                                        phi_35683_ = true;
                                                        break;
                                                    }
                                                    let _e6524_ = local_14_[_e6493_.member_1_];
                                                    if _e6520_ {
                                                    } else {
                                                        phi_35683_ = true;
                                                        break;
                                                    }
                                                    let _e6526_ = local_17_[_e6493_.member_1_];
                                                    if _e6520_ {
                                                    } else {
                                                        phi_35683_ = true;
                                                        break;
                                                    }
                                                    let _e6528_ = local_18_[_e6493_.member_1_];
                                                    if _e6520_ {
                                                    } else {
                                                        phi_35683_ = true;
                                                        break;
                                                    }
                                                    local_9_[_e6493_.member_1_] = _e6526_;
                                                    if _e6520_ {
                                                    } else {
                                                        phi_35683_ = true;
                                                        break;
                                                    }
                                                    local_10_[_e6493_.member_1_] = _e6528_;
                                                    let _e6532_ = (1f - (_e6505_ * _e6526_));
                                                    if (_e6532_ != _e6532_) {
                                                        phi_29836_ = true;
                                                    } else {
                                                        phi_29836_ = (0.05f >= _e6532_);
                                                    }
                                                    let _e6536_ = phi_29836_;
                                                    let _e6539_ = (1f - (_e6505_ * _e6528_));
                                                    if (_e6539_ != _e6539_) {
                                                        phi_29851_ = true;
                                                    } else {
                                                        phi_29851_ = (0.05f >= _e6539_);
                                                    }
                                                    let _e6543_ = phi_29851_;
                                                    let _e6549_ = bitcast<u32>((((_e6522_ * _e6512_) * select(_e6532_, 0.05f, _e6536_)) + (_e6519_ * _e6526_)));
                                                    let _e6550_ = (_e6549_ & 2139095040u);
                                                    if (_e6550_ == 2139095040u) {
                                                        phi_18855_ = true;
                                                    } else {
                                                        phi_18855_ = (_e6550_ == 0u);
                                                    }
                                                    let _e6554_ = phi_18855_;
                                                    let _e6556_ = bitcast<f32>(select(_e6549_, 0u, _e6554_));
                                                    let _e6561_ = bitcast<u32>((((_e6524_ * _e6512_) * select(_e6539_, 0.05f, _e6543_)) + (_e6519_ * _e6528_)));
                                                    let _e6562_ = (_e6561_ & 2139095040u);
                                                    if (_e6562_ == 2139095040u) {
                                                        phi_18875_ = true;
                                                    } else {
                                                        phi_18875_ = (_e6562_ == 0u);
                                                    }
                                                    let _e6566_ = phi_18875_;
                                                    let _e6568_ = bitcast<f32>(select(_e6561_, 0u, _e6566_));
                                                    if _e6520_ {
                                                    } else {
                                                        phi_35683_ = true;
                                                        break;
                                                    }
                                                    local_11_[_e6493_.member_1_] = _e6556_;
                                                    if _e6520_ {
                                                    } else {
                                                        phi_35683_ = true;
                                                        break;
                                                    }
                                                    local_12_[_e6493_.member_1_] = _e6568_;
                                                    let _e6572_ = ((_e6493_.member_1_ * 128u) + _e170_);
                                                    if (_e6572_ < 1536u) {
                                                    } else {
                                                        phi_35683_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e6572_] = _e6556_;
                                                    let _e6577_ = (((4u + _e6493_.member_1_) * 128u) + _e170_);
                                                    if (_e6577_ < 1536u) {
                                                    } else {
                                                        phi_35683_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e6577_] = _e6568_;
                                                }
                                                phi_18331_ = _e6491_;
                                                phi_35683_ = _e6432_;
                                                if !(_e6497_) {
                                                    break;
                                                }
                                            }
                                            let _e6582_ = phi_35683_;
                                            phi_36576_ = _e6582_;
                                            if _e6582_ {
                                                break;
                                            }
                                            phi_35703_ = _e6582_;
                                        } else {
                                            phi_35703_ = _e6377_;
                                        }
                                        let _e6584_ = phi_35703_;
                                        phi_35702_ = _e6584_;
                                    } else {
                                        phi_35702_ = _e6377_;
                                    }
                                    let _e6586_ = phi_35702_;
                                    workgroupBarrier();
                                    if _e6378_ {
                                        if _e1589_ {
                                            local_19_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_20_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            if (_e170_ == 0u) {
                                                phi_18923_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e6628_ = phi_18923_;
                                                    if (_e6628_.member < _e6628_.member_1_) {
                                                        phi_18924_ = type_15_((_e6628_.member + 1u), _e6628_.member_1_);
                                                        phi_18947_ = type_15_(1u, _e6628_.member);
                                                    } else {
                                                        phi_18924_ = _e6628_;
                                                        phi_18947_ = type_15_(0u, u32());
                                                    }
                                                    let _e6641_ = phi_18924_;
                                                    let _e6643_ = phi_18947_;
                                                    let _e6647_ = (bitcast<i32>(_e6643_.member) != 0i);
                                                    if _e6647_ {
                                                        let _e6648_ = (24u + _e6643_.member_1_);
                                                        if (_e6648_ < 256u) {
                                                        } else {
                                                            phi_35704_ = true;
                                                            break;
                                                        }
                                                        let _e6651_ = aux_scratch[_e6648_];
                                                        let _e6652_ = (_e6643_.member_1_ < 4u);
                                                        if _e6652_ {
                                                        } else {
                                                            phi_35704_ = true;
                                                            break;
                                                        }
                                                        local_19_[_e6643_.member_1_] = _e6651_;
                                                        let _e6654_ = (28u + _e6643_.member_1_);
                                                        if (_e6654_ < 256u) {
                                                        } else {
                                                            phi_35704_ = true;
                                                            break;
                                                        }
                                                        let _e6657_ = aux_scratch[_e6654_];
                                                        if _e6652_ {
                                                        } else {
                                                            phi_35704_ = true;
                                                            break;
                                                        }
                                                        local_20_[_e6643_.member_1_] = _e6657_;
                                                    }
                                                    phi_18923_ = _e6641_;
                                                    phi_35704_ = _e6586_;
                                                    if !(_e6647_) {
                                                        break;
                                                    }
                                                }
                                                let _e6661_ = phi_35704_;
                                                phi_36576_ = _e6661_;
                                                if _e6661_ {
                                                    break;
                                                }
                                                phi_36557_ = _e6661_;
                                            } else {
                                                let _e6588_ = (_e170_ - 1u);
                                                phi_18986_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e6590_ = phi_18986_;
                                                    if (_e6590_.member < _e6590_.member_1_) {
                                                        phi_18987_ = type_15_((_e6590_.member + 1u), _e6590_.member_1_);
                                                        phi_19010_ = type_15_(1u, _e6590_.member);
                                                    } else {
                                                        phi_18987_ = _e6590_;
                                                        phi_19010_ = type_15_(0u, u32());
                                                    }
                                                    let _e6603_ = phi_18987_;
                                                    let _e6605_ = phi_19010_;
                                                    let _e6609_ = (bitcast<i32>(_e6605_.member) != 0i);
                                                    if _e6609_ {
                                                        let _e6611_ = ((_e6605_.member_1_ * 128u) + _e6588_);
                                                        if (_e6611_ < 1536u) {
                                                        } else {
                                                            phi_35698_ = true;
                                                            break;
                                                        }
                                                        let _e6614_ = scan_scratch[_e6611_];
                                                        let _e6615_ = (_e6605_.member_1_ < 4u);
                                                        if _e6615_ {
                                                        } else {
                                                            phi_35698_ = true;
                                                            break;
                                                        }
                                                        local_19_[_e6605_.member_1_] = _e6614_;
                                                        let _e6619_ = (((4u + _e6605_.member_1_) * 128u) + _e6588_);
                                                        if (_e6619_ < 1536u) {
                                                        } else {
                                                            phi_35698_ = true;
                                                            break;
                                                        }
                                                        let _e6622_ = scan_scratch[_e6619_];
                                                        if _e6615_ {
                                                        } else {
                                                            phi_35698_ = true;
                                                            break;
                                                        }
                                                        local_20_[_e6605_.member_1_] = _e6622_;
                                                    }
                                                    phi_18986_ = _e6603_;
                                                    phi_35698_ = _e6586_;
                                                    if !(_e6609_) {
                                                        break;
                                                    }
                                                }
                                                let _e6626_ = phi_35698_;
                                                phi_36576_ = _e6626_;
                                                if _e6626_ {
                                                    break;
                                                }
                                                phi_36557_ = _e6626_;
                                            }
                                            let _e6663_ = phi_36557_;
                                            let _e6664_ = local_19_;
                                            let _e6669_ = local_11_;
                                            let _e6674_ = local_20_;
                                            let _e6679_ = local_12_;
                                            let _e6697_ = select(0f, _e6664_[0], (bitcast<f32>((bitcast<u32>(_e6664_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6699_ = select(0f, _e6664_[1], (bitcast<f32>((bitcast<u32>(_e6664_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6701_ = select(0f, _e6664_[2], (bitcast<f32>((bitcast<u32>(_e6664_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6703_ = select(0f, _e6664_[3], (bitcast<f32>((bitcast<u32>(_e6664_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6704_ = (_e6697_ != _e6697_);
                                            if _e6704_ {
                                                phi_29866_ = true;
                                            } else {
                                                phi_29866_ = (-1000000000000000000f >= _e6697_);
                                            }
                                            let _e6707_ = phi_29866_;
                                            let _e6708_ = select(_e6697_, -1000000000000000000f, _e6707_);
                                            let _e6709_ = (_e6699_ != _e6699_);
                                            if _e6709_ {
                                                phi_29881_ = true;
                                            } else {
                                                phi_29881_ = (-1000000000000000000f >= _e6699_);
                                            }
                                            let _e6712_ = phi_29881_;
                                            let _e6713_ = select(_e6699_, -1000000000000000000f, _e6712_);
                                            let _e6714_ = (_e6701_ != _e6701_);
                                            if _e6714_ {
                                                phi_29896_ = true;
                                            } else {
                                                phi_29896_ = (-1000000000000000000f >= _e6701_);
                                            }
                                            let _e6717_ = phi_29896_;
                                            let _e6718_ = select(_e6701_, -1000000000000000000f, _e6717_);
                                            let _e6719_ = (_e6703_ != _e6703_);
                                            if _e6719_ {
                                                phi_29911_ = true;
                                            } else {
                                                phi_29911_ = (-1000000000000000000f >= _e6703_);
                                            }
                                            let _e6722_ = phi_29911_;
                                            let _e6723_ = select(_e6703_, -1000000000000000000f, _e6722_);
                                            if (_e6708_ != _e6708_) {
                                                phi_29926_ = true;
                                            } else {
                                                phi_29926_ = (1000000000000000000f <= _e6708_);
                                            }
                                            let _e6727_ = phi_29926_;
                                            let _e6728_ = select(_e6708_, 1000000000000000000f, _e6727_);
                                            if (_e6713_ != _e6713_) {
                                                phi_29941_ = true;
                                            } else {
                                                phi_29941_ = (1000000000000000000f <= _e6713_);
                                            }
                                            let _e6732_ = phi_29941_;
                                            let _e6733_ = select(_e6713_, 1000000000000000000f, _e6732_);
                                            if (_e6718_ != _e6718_) {
                                                phi_29956_ = true;
                                            } else {
                                                phi_29956_ = (1000000000000000000f <= _e6718_);
                                            }
                                            let _e6737_ = phi_29956_;
                                            let _e6738_ = select(_e6718_, 1000000000000000000f, _e6737_);
                                            if (_e6723_ != _e6723_) {
                                                phi_29971_ = true;
                                            } else {
                                                phi_29971_ = (1000000000000000000f <= _e6723_);
                                            }
                                            let _e6742_ = phi_29971_;
                                            let _e6743_ = select(_e6723_, 1000000000000000000f, _e6742_);
                                            let _e6757_ = select(0f, _e6669_[0], (bitcast<f32>((bitcast<u32>(_e6669_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6759_ = select(0f, _e6669_[1], (bitcast<f32>((bitcast<u32>(_e6669_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6761_ = select(0f, _e6669_[2], (bitcast<f32>((bitcast<u32>(_e6669_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6763_ = select(0f, _e6669_[3], (bitcast<f32>((bitcast<u32>(_e6669_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6764_ = (_e6757_ != _e6757_);
                                            if _e6764_ {
                                                phi_29986_ = true;
                                            } else {
                                                phi_29986_ = (-1000000000000000000f >= _e6757_);
                                            }
                                            let _e6767_ = phi_29986_;
                                            let _e6768_ = select(_e6757_, -1000000000000000000f, _e6767_);
                                            let _e6769_ = (_e6759_ != _e6759_);
                                            if _e6769_ {
                                                phi_30001_ = true;
                                            } else {
                                                phi_30001_ = (-1000000000000000000f >= _e6759_);
                                            }
                                            let _e6772_ = phi_30001_;
                                            let _e6773_ = select(_e6759_, -1000000000000000000f, _e6772_);
                                            let _e6774_ = (_e6761_ != _e6761_);
                                            if _e6774_ {
                                                phi_30016_ = true;
                                            } else {
                                                phi_30016_ = (-1000000000000000000f >= _e6761_);
                                            }
                                            let _e6777_ = phi_30016_;
                                            let _e6778_ = select(_e6761_, -1000000000000000000f, _e6777_);
                                            let _e6779_ = (_e6763_ != _e6763_);
                                            if _e6779_ {
                                                phi_30031_ = true;
                                            } else {
                                                phi_30031_ = (-1000000000000000000f >= _e6763_);
                                            }
                                            let _e6782_ = phi_30031_;
                                            let _e6783_ = select(_e6763_, -1000000000000000000f, _e6782_);
                                            if (_e6768_ != _e6768_) {
                                                phi_30046_ = true;
                                            } else {
                                                phi_30046_ = (1000000000000000000f <= _e6768_);
                                            }
                                            let _e6787_ = phi_30046_;
                                            let _e6788_ = select(_e6768_, 1000000000000000000f, _e6787_);
                                            if (_e6773_ != _e6773_) {
                                                phi_30061_ = true;
                                            } else {
                                                phi_30061_ = (1000000000000000000f <= _e6773_);
                                            }
                                            let _e6792_ = phi_30061_;
                                            let _e6793_ = select(_e6773_, 1000000000000000000f, _e6792_);
                                            if (_e6778_ != _e6778_) {
                                                phi_30076_ = true;
                                            } else {
                                                phi_30076_ = (1000000000000000000f <= _e6778_);
                                            }
                                            let _e6797_ = phi_30076_;
                                            let _e6798_ = select(_e6778_, 1000000000000000000f, _e6797_);
                                            if (_e6783_ != _e6783_) {
                                                phi_30091_ = true;
                                            } else {
                                                phi_30091_ = (1000000000000000000f <= _e6783_);
                                            }
                                            let _e6802_ = phi_30091_;
                                            let _e6803_ = select(_e6783_, 1000000000000000000f, _e6802_);
                                            let _e6804_ = array<f32, 4>(_e6788_, _e6793_, _e6798_, _e6803_);
                                            let _e6805_ = (_e6728_ != _e6728_);
                                            if _e6805_ {
                                                phi_30106_ = true;
                                            } else {
                                                phi_30106_ = (_e6788_ <= _e6728_);
                                            }
                                            let _e6808_ = phi_30106_;
                                            let _e6809_ = select(_e6728_, _e6788_, _e6808_);
                                            let _e6810_ = (_e6733_ != _e6733_);
                                            if _e6810_ {
                                                phi_30121_ = true;
                                            } else {
                                                phi_30121_ = (_e6793_ <= _e6733_);
                                            }
                                            let _e6813_ = phi_30121_;
                                            let _e6814_ = select(_e6733_, _e6793_, _e6813_);
                                            let _e6815_ = (_e6738_ != _e6738_);
                                            if _e6815_ {
                                                phi_30136_ = true;
                                            } else {
                                                phi_30136_ = (_e6798_ <= _e6738_);
                                            }
                                            let _e6818_ = phi_30136_;
                                            let _e6819_ = select(_e6738_, _e6798_, _e6818_);
                                            let _e6820_ = (_e6743_ != _e6743_);
                                            if _e6820_ {
                                                phi_30151_ = true;
                                            } else {
                                                phi_30151_ = (_e6803_ <= _e6743_);
                                            }
                                            let _e6823_ = phi_30151_;
                                            let _e6824_ = select(_e6743_, _e6803_, _e6823_);
                                            if _e6805_ {
                                                phi_30166_ = true;
                                            } else {
                                                phi_30166_ = (_e6788_ >= _e6728_);
                                            }
                                            let _e6827_ = phi_30166_;
                                            let _e6828_ = select(_e6728_, _e6788_, _e6827_);
                                            if _e6810_ {
                                                phi_30181_ = true;
                                            } else {
                                                phi_30181_ = (_e6793_ >= _e6733_);
                                            }
                                            let _e6831_ = phi_30181_;
                                            let _e6832_ = select(_e6733_, _e6793_, _e6831_);
                                            if _e6815_ {
                                                phi_30196_ = true;
                                            } else {
                                                phi_30196_ = (_e6798_ >= _e6738_);
                                            }
                                            let _e6835_ = phi_30196_;
                                            let _e6836_ = select(_e6738_, _e6798_, _e6835_);
                                            if _e6820_ {
                                                phi_30211_ = true;
                                            } else {
                                                phi_30211_ = (_e6803_ >= _e6743_);
                                            }
                                            let _e6839_ = phi_30211_;
                                            let _e6840_ = select(_e6743_, _e6803_, _e6839_);
                                            let _e6841_ = (_e6809_ != _e6809_);
                                            if _e6841_ {
                                                phi_30226_ = true;
                                            } else {
                                                phi_30226_ = (-1f >= _e6809_);
                                            }
                                            let _e6844_ = phi_30226_;
                                            let _e6845_ = select(_e6809_, -1f, _e6844_);
                                            let _e6846_ = (_e6814_ != _e6814_);
                                            if _e6846_ {
                                                phi_30241_ = true;
                                            } else {
                                                phi_30241_ = (-1f >= _e6814_);
                                            }
                                            let _e6849_ = phi_30241_;
                                            let _e6850_ = select(_e6814_, -1f, _e6849_);
                                            let _e6851_ = (_e6819_ != _e6819_);
                                            if _e6851_ {
                                                phi_30256_ = true;
                                            } else {
                                                phi_30256_ = (-1f >= _e6819_);
                                            }
                                            let _e6854_ = phi_30256_;
                                            let _e6855_ = select(_e6819_, -1f, _e6854_);
                                            let _e6856_ = (_e6824_ != _e6824_);
                                            if _e6856_ {
                                                phi_30271_ = true;
                                            } else {
                                                phi_30271_ = (-1f >= _e6824_);
                                            }
                                            let _e6859_ = phi_30271_;
                                            let _e6860_ = select(_e6824_, -1f, _e6859_);
                                            if (_e6845_ != _e6845_) {
                                                phi_30286_ = true;
                                            } else {
                                                phi_30286_ = (1f <= _e6845_);
                                            }
                                            let _e6864_ = phi_30286_;
                                            let _e6865_ = select(_e6845_, 1f, _e6864_);
                                            if (_e6850_ != _e6850_) {
                                                phi_30301_ = true;
                                            } else {
                                                phi_30301_ = (1f <= _e6850_);
                                            }
                                            let _e6869_ = phi_30301_;
                                            let _e6870_ = select(_e6850_, 1f, _e6869_);
                                            if (_e6855_ != _e6855_) {
                                                phi_30316_ = true;
                                            } else {
                                                phi_30316_ = (1f <= _e6855_);
                                            }
                                            let _e6874_ = phi_30316_;
                                            let _e6875_ = select(_e6855_, 1f, _e6874_);
                                            if (_e6860_ != _e6860_) {
                                                phi_30331_ = true;
                                            } else {
                                                phi_30331_ = (1f <= _e6860_);
                                            }
                                            let _e6879_ = phi_30331_;
                                            let _e6880_ = select(_e6860_, 1f, _e6879_);
                                            let _e6881_ = (_e6828_ != _e6828_);
                                            if _e6881_ {
                                                phi_30346_ = true;
                                            } else {
                                                phi_30346_ = (-1f >= _e6828_);
                                            }
                                            let _e6884_ = phi_30346_;
                                            let _e6885_ = select(_e6828_, -1f, _e6884_);
                                            let _e6886_ = (_e6832_ != _e6832_);
                                            if _e6886_ {
                                                phi_30361_ = true;
                                            } else {
                                                phi_30361_ = (-1f >= _e6832_);
                                            }
                                            let _e6889_ = phi_30361_;
                                            let _e6890_ = select(_e6832_, -1f, _e6889_);
                                            let _e6891_ = (_e6836_ != _e6836_);
                                            if _e6891_ {
                                                phi_30376_ = true;
                                            } else {
                                                phi_30376_ = (-1f >= _e6836_);
                                            }
                                            let _e6894_ = phi_30376_;
                                            let _e6895_ = select(_e6836_, -1f, _e6894_);
                                            let _e6896_ = (_e6840_ != _e6840_);
                                            if _e6896_ {
                                                phi_30391_ = true;
                                            } else {
                                                phi_30391_ = (-1f >= _e6840_);
                                            }
                                            let _e6899_ = phi_30391_;
                                            let _e6900_ = select(_e6840_, -1f, _e6899_);
                                            if (_e6885_ != _e6885_) {
                                                phi_30406_ = true;
                                            } else {
                                                phi_30406_ = (1f <= _e6885_);
                                            }
                                            let _e6904_ = phi_30406_;
                                            let _e6905_ = select(_e6885_, 1f, _e6904_);
                                            if (_e6890_ != _e6890_) {
                                                phi_30421_ = true;
                                            } else {
                                                phi_30421_ = (1f <= _e6890_);
                                            }
                                            let _e6909_ = phi_30421_;
                                            let _e6910_ = select(_e6890_, 1f, _e6909_);
                                            if (_e6895_ != _e6895_) {
                                                phi_30436_ = true;
                                            } else {
                                                phi_30436_ = (1f <= _e6895_);
                                            }
                                            let _e6914_ = phi_30436_;
                                            let _e6915_ = select(_e6895_, 1f, _e6914_);
                                            if (_e6900_ != _e6900_) {
                                                phi_30451_ = true;
                                            } else {
                                                phi_30451_ = (1f <= _e6900_);
                                            }
                                            let _e6919_ = phi_30451_;
                                            let _e6920_ = select(_e6900_, 1f, _e6919_);
                                            let _e6925_ = (0.5f * (_e6905_ + _e6865_));
                                            let _e6926_ = (0.5f * (_e6910_ + _e6870_));
                                            let _e6927_ = (0.5f * (_e6915_ + _e6875_));
                                            let _e6928_ = (0.5f * (_e6920_ + _e6880_));
                                            let _e6929_ = (_e6905_ - _e6865_);
                                            let _e6930_ = (_e6910_ - _e6870_);
                                            let _e6931_ = (_e6915_ - _e6875_);
                                            let _e6932_ = (_e6920_ - _e6880_);
                                            let _e6933_ = (0.5f * _e6929_);
                                            let _e6934_ = (0.5f * _e6930_);
                                            let _e6935_ = (0.5f * _e6931_);
                                            let _e6936_ = (0.5f * _e6932_);
                                            let _e6937_ = (_e6925_ * _e6925_);
                                            let _e6938_ = (_e6926_ * _e6926_);
                                            let _e6939_ = (_e6927_ * _e6927_);
                                            let _e6940_ = (_e6928_ * _e6928_);
                                            let _e6941_ = (_e6933_ * _e6933_);
                                            let _e6942_ = (_e6934_ * _e6934_);
                                            let _e6943_ = (_e6935_ * _e6935_);
                                            let _e6944_ = (_e6936_ * _e6936_);
                                            if (_e6929_ != _e6929_) {
                                                phi_30466_ = true;
                                            } else {
                                                phi_30466_ = (0f >= _e6929_);
                                            }
                                            let _e6948_ = phi_30466_;
                                            let _e6949_ = select(_e6929_, 0f, _e6948_);
                                            if (_e6930_ != _e6930_) {
                                                phi_30481_ = true;
                                            } else {
                                                phi_30481_ = (0f >= _e6930_);
                                            }
                                            let _e6953_ = phi_30481_;
                                            let _e6954_ = select(_e6930_, 0f, _e6953_);
                                            if (_e6931_ != _e6931_) {
                                                phi_30496_ = true;
                                            } else {
                                                phi_30496_ = (0f >= _e6931_);
                                            }
                                            let _e6958_ = phi_30496_;
                                            let _e6959_ = select(_e6931_, 0f, _e6958_);
                                            if (_e6932_ != _e6932_) {
                                                phi_30511_ = true;
                                            } else {
                                                phi_30511_ = (0f >= _e6932_);
                                            }
                                            let _e6963_ = phi_30511_;
                                            let _e6964_ = select(_e6932_, 0f, _e6963_);
                                            let _e6965_ = (_e6828_ - _e6809_);
                                            let _e6966_ = (_e6832_ - _e6814_);
                                            let _e6967_ = (_e6836_ - _e6819_);
                                            let _e6968_ = (_e6840_ - _e6824_);
                                            if _e6881_ {
                                                phi_30526_ = true;
                                            } else {
                                                phi_30526_ = (-1f <= _e6828_);
                                            }
                                            let _e6971_ = phi_30526_;
                                            if _e6886_ {
                                                phi_30541_ = true;
                                            } else {
                                                phi_30541_ = (-1f <= _e6832_);
                                            }
                                            let _e6975_ = phi_30541_;
                                            if _e6891_ {
                                                phi_30556_ = true;
                                            } else {
                                                phi_30556_ = (-1f <= _e6836_);
                                            }
                                            let _e6979_ = phi_30556_;
                                            if _e6896_ {
                                                phi_30571_ = true;
                                            } else {
                                                phi_30571_ = (-1f <= _e6840_);
                                            }
                                            let _e6983_ = phi_30571_;
                                            let _e6985_ = (select(_e6828_, -1f, _e6971_) - _e6809_);
                                            let _e6986_ = (select(_e6832_, -1f, _e6975_) - _e6814_);
                                            let _e6987_ = (select(_e6836_, -1f, _e6979_) - _e6819_);
                                            let _e6988_ = (select(_e6840_, -1f, _e6983_) - _e6824_);
                                            if (_e6985_ != _e6985_) {
                                                phi_30586_ = true;
                                            } else {
                                                phi_30586_ = (0f >= _e6985_);
                                            }
                                            let _e6992_ = phi_30586_;
                                            if (_e6986_ != _e6986_) {
                                                phi_30601_ = true;
                                            } else {
                                                phi_30601_ = (0f >= _e6986_);
                                            }
                                            let _e6997_ = phi_30601_;
                                            if (_e6987_ != _e6987_) {
                                                phi_30616_ = true;
                                            } else {
                                                phi_30616_ = (0f >= _e6987_);
                                            }
                                            let _e7002_ = phi_30616_;
                                            if (_e6988_ != _e6988_) {
                                                phi_30631_ = true;
                                            } else {
                                                phi_30631_ = (0f >= _e6988_);
                                            }
                                            let _e7007_ = phi_30631_;
                                            if _e6841_ {
                                                phi_30646_ = true;
                                            } else {
                                                phi_30646_ = (1f >= _e6809_);
                                            }
                                            let _e7011_ = phi_30646_;
                                            if _e6846_ {
                                                phi_30661_ = true;
                                            } else {
                                                phi_30661_ = (1f >= _e6814_);
                                            }
                                            let _e7015_ = phi_30661_;
                                            if _e6851_ {
                                                phi_30676_ = true;
                                            } else {
                                                phi_30676_ = (1f >= _e6819_);
                                            }
                                            let _e7019_ = phi_30676_;
                                            if _e6856_ {
                                                phi_30691_ = true;
                                            } else {
                                                phi_30691_ = (1f >= _e6824_);
                                            }
                                            let _e7023_ = phi_30691_;
                                            let _e7025_ = (_e6828_ - select(_e6809_, 1f, _e7011_));
                                            let _e7026_ = (_e6832_ - select(_e6814_, 1f, _e7015_));
                                            let _e7027_ = (_e6836_ - select(_e6819_, 1f, _e7019_));
                                            let _e7028_ = (_e6840_ - select(_e6824_, 1f, _e7023_));
                                            if (_e7025_ != _e7025_) {
                                                phi_30706_ = true;
                                            } else {
                                                phi_30706_ = (0f >= _e7025_);
                                            }
                                            let _e7032_ = phi_30706_;
                                            if (_e7026_ != _e7026_) {
                                                phi_30721_ = true;
                                            } else {
                                                phi_30721_ = (0f >= _e7026_);
                                            }
                                            let _e7037_ = phi_30721_;
                                            if (_e7027_ != _e7027_) {
                                                phi_30736_ = true;
                                            } else {
                                                phi_30736_ = (0f >= _e7027_);
                                            }
                                            let _e7042_ = phi_30736_;
                                            if (_e7028_ != _e7028_) {
                                                phi_30751_ = true;
                                            } else {
                                                phi_30751_ = (0f >= _e7028_);
                                            }
                                            let _e7047_ = phi_30751_;
                                            let _e7051_ = bitcast<f32>((bitcast<u32>(_e6728_) & 2147483647u));
                                            let _e7054_ = bitcast<f32>((bitcast<u32>(_e6733_) & 2147483647u));
                                            let _e7057_ = bitcast<f32>((bitcast<u32>(_e6738_) & 2147483647u));
                                            let _e7060_ = bitcast<f32>((bitcast<u32>(_e6743_) & 2147483647u));
                                            let _e7063_ = bitcast<f32>((bitcast<u32>(_e6788_) & 2147483647u));
                                            let _e7066_ = bitcast<f32>((bitcast<u32>(_e6793_) & 2147483647u));
                                            let _e7069_ = bitcast<f32>((bitcast<u32>(_e6798_) & 2147483647u));
                                            let _e7072_ = bitcast<f32>((bitcast<u32>(_e6803_) & 2147483647u));
                                            if (_e7051_ != _e7051_) {
                                                phi_30766_ = true;
                                            } else {
                                                phi_30766_ = (_e7063_ >= _e7051_);
                                            }
                                            let _e7076_ = phi_30766_;
                                            let _e7077_ = select(_e7051_, _e7063_, _e7076_);
                                            if (_e7054_ != _e7054_) {
                                                phi_30781_ = true;
                                            } else {
                                                phi_30781_ = (_e7066_ >= _e7054_);
                                            }
                                            let _e7081_ = phi_30781_;
                                            let _e7082_ = select(_e7054_, _e7066_, _e7081_);
                                            if (_e7057_ != _e7057_) {
                                                phi_30796_ = true;
                                            } else {
                                                phi_30796_ = (_e7069_ >= _e7057_);
                                            }
                                            let _e7086_ = phi_30796_;
                                            let _e7087_ = select(_e7057_, _e7069_, _e7086_);
                                            if (_e7060_ != _e7060_) {
                                                phi_30811_ = true;
                                            } else {
                                                phi_30811_ = (_e7072_ >= _e7060_);
                                            }
                                            let _e7091_ = phi_30811_;
                                            let _e7092_ = select(_e7060_, _e7072_, _e7091_);
                                            let _e7109_ = (_e7077_ <= 0f);
                                            let _e7110_ = select((_e6925_ * (1.5f - (0.5f * (_e6937_ + _e6941_)))), 0f, _e7109_);
                                            let _e7111_ = (_e7082_ <= 0f);
                                            let _e7112_ = select((_e6926_ * (1.5f - (0.5f * (_e6938_ + _e6942_)))), 0f, _e7111_);
                                            let _e7113_ = (_e7087_ <= 0f);
                                            let _e7114_ = select((_e6927_ * (1.5f - (0.5f * (_e6939_ + _e6943_)))), 0f, _e7113_);
                                            let _e7115_ = (_e7092_ <= 0f);
                                            let _e7116_ = select((_e6928_ * (1.5f - (0.5f * (_e6940_ + _e6944_)))), 0f, _e7115_);
                                            let _e7158_ = select(((_e6937_ + (0.33333334f * _e6941_)) - (((_e6937_ * _e6937_) + ((2f * _e6937_) * _e6941_)) + (0.2f * (_e6941_ * _e6941_)))), 0f, _e7109_);
                                            let _e7159_ = select(((_e6938_ + (0.33333334f * _e6942_)) - (((_e6938_ * _e6938_) + ((2f * _e6938_) * _e6942_)) + (0.2f * (_e6942_ * _e6942_)))), 0f, _e7111_);
                                            let _e7160_ = select(((_e6939_ + (0.33333334f * _e6943_)) - (((_e6939_ * _e6939_) + ((2f * _e6939_) * _e6943_)) + (0.2f * (_e6943_ * _e6943_)))), 0f, _e7113_);
                                            let _e7161_ = select(((_e6940_ + (0.33333334f * _e6944_)) - (((_e6940_ * _e6940_) + ((2f * _e6940_) * _e6944_)) + (0.2f * (_e6944_ * _e6944_)))), 0f, _e7115_);
                                            if (_e7077_ <= 1f) {
                                                if (_e7082_ <= 1f) {
                                                    let _e7165_ = (_e7087_ <= 1f);
                                                    if _e7165_ {
                                                        phi_19515_ = (_e7092_ <= 1f);
                                                    } else {
                                                        phi_19515_ = bool();
                                                    }
                                                    let _e7168_ = phi_19515_;
                                                    phi_19517_ = _e7168_;
                                                    phi_19518_ = select(true, false, _e7165_);
                                                } else {
                                                    phi_19517_ = bool();
                                                    phi_19518_ = true;
                                                }
                                                let _e7171_ = phi_19517_;
                                                let _e7173_ = phi_19518_;
                                                phi_19519_ = _e7171_;
                                                phi_19520_ = _e7173_;
                                            } else {
                                                phi_19519_ = bool();
                                                phi_19520_ = true;
                                            }
                                            let _e7175_ = phi_19519_;
                                            let _e7177_ = phi_19520_;
                                            if select(_e7175_, false, _e7177_) {
                                                phi_19898_ = type_38_(type_30_(array<f32, 4>(_e7110_, _e7112_, _e7114_, _e7116_)), type_30_(array<f32, 4>(_e7158_, _e7159_, _e7160_, _e7161_)), type_30_(_e6804_));
                                            } else {
                                                if (_e6965_ != _e6965_) {
                                                    phi_30826_ = true;
                                                } else {
                                                    phi_30826_ = (0.000001f >= _e6965_);
                                                }
                                                let _e7182_ = phi_30826_;
                                                let _e7183_ = select(_e6965_, 0.000001f, _e7182_);
                                                if (_e6966_ != _e6966_) {
                                                    phi_30841_ = true;
                                                } else {
                                                    phi_30841_ = (0.000001f >= _e6966_);
                                                }
                                                let _e7187_ = phi_30841_;
                                                let _e7188_ = select(_e6966_, 0.000001f, _e7187_);
                                                if (_e6967_ != _e6967_) {
                                                    phi_30856_ = true;
                                                } else {
                                                    phi_30856_ = (0.000001f >= _e6967_);
                                                }
                                                let _e7192_ = phi_30856_;
                                                let _e7193_ = select(_e6967_, 0.000001f, _e7192_);
                                                if (_e6968_ != _e6968_) {
                                                    phi_30871_ = true;
                                                } else {
                                                    phi_30871_ = (0.000001f >= _e6968_);
                                                }
                                                let _e7197_ = phi_30871_;
                                                let _e7198_ = select(_e6968_, 0.000001f, _e7197_);
                                                let _e7201_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7183_)));
                                                let _e7204_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7188_)));
                                                let _e7207_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7193_)));
                                                let _e7210_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7198_)));
                                                let _e7220_ = (_e7201_ + (_e7201_ * (1f - (_e7183_ * _e7201_))));
                                                let _e7222_ = (_e7204_ + (_e7204_ * (1f - (_e7188_ * _e7204_))));
                                                let _e7224_ = (_e7207_ + (_e7207_ * (1f - (_e7193_ * _e7207_))));
                                                let _e7226_ = (_e7210_ + (_e7210_ * (1f - (_e7198_ * _e7210_))));
                                                let _e7236_ = (_e7220_ + (_e7220_ * (1f - (_e7183_ * _e7220_))));
                                                let _e7238_ = (_e7222_ + (_e7222_ * (1f - (_e7188_ * _e7222_))));
                                                let _e7240_ = (_e7224_ + (_e7224_ * (1f - (_e7193_ * _e7224_))));
                                                let _e7242_ = (_e7226_ + (_e7226_ * (1f - (_e7198_ * _e7226_))));
                                                let _e7295_ = (select((select(0f, 1f, (-1f <= _e6809_)) * select(0f, 1f, (_e6828_ <= 1f))), 1f, (_e6965_ <= 0.000001f)) <= 0f);
                                                let _e7297_ = (select((select(0f, 1f, (-1f <= _e6814_)) * select(0f, 1f, (_e6832_ <= 1f))), 1f, (_e6966_ <= 0.000001f)) <= 0f);
                                                let _e7299_ = (select((select(0f, 1f, (-1f <= _e6819_)) * select(0f, 1f, (_e6836_ <= 1f))), 1f, (_e6967_ <= 0.000001f)) <= 0f);
                                                let _e7301_ = (select((select(0f, 1f, (-1f <= _e6824_)) * select(0f, 1f, (_e6840_ <= 1f))), 1f, (_e6968_ <= 0.000001f)) <= 0f);
                                                let _e7307_ = (1f <= _e6809_);
                                                let _e7309_ = (1f <= _e6814_);
                                                let _e7311_ = (1f <= _e6819_);
                                                let _e7313_ = (1f <= _e6824_);
                                                let _e7315_ = (_e6828_ <= -1f);
                                                let _e7317_ = (_e6832_ <= -1f);
                                                let _e7319_ = (_e6836_ <= -1f);
                                                let _e7321_ = (_e6840_ <= -1f);
                                                phi_19898_ = type_38_(type_30_(array<f32, 4>(select(select(select(_e7110_, (((select(_e7025_, 0f, _e7032_) - select(_e6985_, 0f, _e6992_)) + (_e6949_ * _e7110_)) * _e7236_), _e7295_), 1f, _e7307_), -1f, _e7315_), select(select(select(_e7112_, (((select(_e7026_, 0f, _e7037_) - select(_e6986_, 0f, _e6997_)) + (_e6954_ * _e7112_)) * _e7238_), _e7297_), 1f, _e7309_), -1f, _e7317_), select(select(select(_e7114_, (((select(_e7027_, 0f, _e7042_) - select(_e6987_, 0f, _e7002_)) + (_e6959_ * _e7114_)) * _e7240_), _e7299_), 1f, _e7311_), -1f, _e7319_), select(select(select(_e7116_, (((select(_e7028_, 0f, _e7047_) - select(_e6988_, 0f, _e7007_)) + (_e6964_ * _e7116_)) * _e7242_), _e7301_), 1f, _e7313_), -1f, _e7321_))), type_30_(array<f32, 4>(select(select(select(_e7158_, ((_e6949_ * _e7158_) * _e7236_), _e7295_), 0f, _e7307_), 0f, _e7315_), select(select(select(_e7159_, ((_e6954_ * _e7159_) * _e7238_), _e7297_), 0f, _e7309_), 0f, _e7317_), select(select(select(_e7160_, ((_e6959_ * _e7160_) * _e7240_), _e7299_), 0f, _e7311_), 0f, _e7319_), select(select(select(_e7161_, ((_e6964_ * _e7161_) * _e7242_), _e7301_), 0f, _e7313_), 0f, _e7321_))), type_30_(_e6804_));
                                            }
                                            let _e7342_ = phi_19898_;
                                            let _e7357_ = select(0f, _e6674_[0], (bitcast<f32>((bitcast<u32>(_e6674_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7359_ = select(0f, _e6674_[1], (bitcast<f32>((bitcast<u32>(_e6674_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7361_ = select(0f, _e6674_[2], (bitcast<f32>((bitcast<u32>(_e6674_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7363_ = select(0f, _e6674_[3], (bitcast<f32>((bitcast<u32>(_e6674_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7364_ = (_e7357_ != _e7357_);
                                            if _e7364_ {
                                                phi_30886_ = true;
                                            } else {
                                                phi_30886_ = (-1000000000000000000f >= _e7357_);
                                            }
                                            let _e7367_ = phi_30886_;
                                            let _e7368_ = select(_e7357_, -1000000000000000000f, _e7367_);
                                            let _e7369_ = (_e7359_ != _e7359_);
                                            if _e7369_ {
                                                phi_30901_ = true;
                                            } else {
                                                phi_30901_ = (-1000000000000000000f >= _e7359_);
                                            }
                                            let _e7372_ = phi_30901_;
                                            let _e7373_ = select(_e7359_, -1000000000000000000f, _e7372_);
                                            let _e7374_ = (_e7361_ != _e7361_);
                                            if _e7374_ {
                                                phi_30916_ = true;
                                            } else {
                                                phi_30916_ = (-1000000000000000000f >= _e7361_);
                                            }
                                            let _e7377_ = phi_30916_;
                                            let _e7378_ = select(_e7361_, -1000000000000000000f, _e7377_);
                                            let _e7379_ = (_e7363_ != _e7363_);
                                            if _e7379_ {
                                                phi_30931_ = true;
                                            } else {
                                                phi_30931_ = (-1000000000000000000f >= _e7363_);
                                            }
                                            let _e7382_ = phi_30931_;
                                            let _e7383_ = select(_e7363_, -1000000000000000000f, _e7382_);
                                            if (_e7368_ != _e7368_) {
                                                phi_30946_ = true;
                                            } else {
                                                phi_30946_ = (1000000000000000000f <= _e7368_);
                                            }
                                            let _e7387_ = phi_30946_;
                                            let _e7388_ = select(_e7368_, 1000000000000000000f, _e7387_);
                                            if (_e7373_ != _e7373_) {
                                                phi_30961_ = true;
                                            } else {
                                                phi_30961_ = (1000000000000000000f <= _e7373_);
                                            }
                                            let _e7392_ = phi_30961_;
                                            let _e7393_ = select(_e7373_, 1000000000000000000f, _e7392_);
                                            if (_e7378_ != _e7378_) {
                                                phi_30976_ = true;
                                            } else {
                                                phi_30976_ = (1000000000000000000f <= _e7378_);
                                            }
                                            let _e7397_ = phi_30976_;
                                            let _e7398_ = select(_e7378_, 1000000000000000000f, _e7397_);
                                            if (_e7383_ != _e7383_) {
                                                phi_30991_ = true;
                                            } else {
                                                phi_30991_ = (1000000000000000000f <= _e7383_);
                                            }
                                            let _e7402_ = phi_30991_;
                                            let _e7403_ = select(_e7383_, 1000000000000000000f, _e7402_);
                                            let _e7417_ = select(0f, _e6679_[0], (bitcast<f32>((bitcast<u32>(_e6679_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7419_ = select(0f, _e6679_[1], (bitcast<f32>((bitcast<u32>(_e6679_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7421_ = select(0f, _e6679_[2], (bitcast<f32>((bitcast<u32>(_e6679_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7423_ = select(0f, _e6679_[3], (bitcast<f32>((bitcast<u32>(_e6679_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7424_ = (_e7417_ != _e7417_);
                                            if _e7424_ {
                                                phi_31006_ = true;
                                            } else {
                                                phi_31006_ = (-1000000000000000000f >= _e7417_);
                                            }
                                            let _e7427_ = phi_31006_;
                                            let _e7428_ = select(_e7417_, -1000000000000000000f, _e7427_);
                                            let _e7429_ = (_e7419_ != _e7419_);
                                            if _e7429_ {
                                                phi_31021_ = true;
                                            } else {
                                                phi_31021_ = (-1000000000000000000f >= _e7419_);
                                            }
                                            let _e7432_ = phi_31021_;
                                            let _e7433_ = select(_e7419_, -1000000000000000000f, _e7432_);
                                            let _e7434_ = (_e7421_ != _e7421_);
                                            if _e7434_ {
                                                phi_31036_ = true;
                                            } else {
                                                phi_31036_ = (-1000000000000000000f >= _e7421_);
                                            }
                                            let _e7437_ = phi_31036_;
                                            let _e7438_ = select(_e7421_, -1000000000000000000f, _e7437_);
                                            let _e7439_ = (_e7423_ != _e7423_);
                                            if _e7439_ {
                                                phi_31051_ = true;
                                            } else {
                                                phi_31051_ = (-1000000000000000000f >= _e7423_);
                                            }
                                            let _e7442_ = phi_31051_;
                                            let _e7443_ = select(_e7423_, -1000000000000000000f, _e7442_);
                                            if (_e7428_ != _e7428_) {
                                                phi_31066_ = true;
                                            } else {
                                                phi_31066_ = (1000000000000000000f <= _e7428_);
                                            }
                                            let _e7447_ = phi_31066_;
                                            let _e7448_ = select(_e7428_, 1000000000000000000f, _e7447_);
                                            if (_e7433_ != _e7433_) {
                                                phi_31081_ = true;
                                            } else {
                                                phi_31081_ = (1000000000000000000f <= _e7433_);
                                            }
                                            let _e7452_ = phi_31081_;
                                            let _e7453_ = select(_e7433_, 1000000000000000000f, _e7452_);
                                            if (_e7438_ != _e7438_) {
                                                phi_31096_ = true;
                                            } else {
                                                phi_31096_ = (1000000000000000000f <= _e7438_);
                                            }
                                            let _e7457_ = phi_31096_;
                                            let _e7458_ = select(_e7438_, 1000000000000000000f, _e7457_);
                                            if (_e7443_ != _e7443_) {
                                                phi_31111_ = true;
                                            } else {
                                                phi_31111_ = (1000000000000000000f <= _e7443_);
                                            }
                                            let _e7462_ = phi_31111_;
                                            let _e7463_ = select(_e7443_, 1000000000000000000f, _e7462_);
                                            let _e7464_ = array<f32, 4>(_e7448_, _e7453_, _e7458_, _e7463_);
                                            let _e7465_ = (_e7388_ != _e7388_);
                                            if _e7465_ {
                                                phi_31126_ = true;
                                            } else {
                                                phi_31126_ = (_e7448_ <= _e7388_);
                                            }
                                            let _e7468_ = phi_31126_;
                                            let _e7469_ = select(_e7388_, _e7448_, _e7468_);
                                            let _e7470_ = (_e7393_ != _e7393_);
                                            if _e7470_ {
                                                phi_31141_ = true;
                                            } else {
                                                phi_31141_ = (_e7453_ <= _e7393_);
                                            }
                                            let _e7473_ = phi_31141_;
                                            let _e7474_ = select(_e7393_, _e7453_, _e7473_);
                                            let _e7475_ = (_e7398_ != _e7398_);
                                            if _e7475_ {
                                                phi_31156_ = true;
                                            } else {
                                                phi_31156_ = (_e7458_ <= _e7398_);
                                            }
                                            let _e7478_ = phi_31156_;
                                            let _e7479_ = select(_e7398_, _e7458_, _e7478_);
                                            let _e7480_ = (_e7403_ != _e7403_);
                                            if _e7480_ {
                                                phi_31171_ = true;
                                            } else {
                                                phi_31171_ = (_e7463_ <= _e7403_);
                                            }
                                            let _e7483_ = phi_31171_;
                                            let _e7484_ = select(_e7403_, _e7463_, _e7483_);
                                            if _e7465_ {
                                                phi_31186_ = true;
                                            } else {
                                                phi_31186_ = (_e7448_ >= _e7388_);
                                            }
                                            let _e7487_ = phi_31186_;
                                            let _e7488_ = select(_e7388_, _e7448_, _e7487_);
                                            if _e7470_ {
                                                phi_31201_ = true;
                                            } else {
                                                phi_31201_ = (_e7453_ >= _e7393_);
                                            }
                                            let _e7491_ = phi_31201_;
                                            let _e7492_ = select(_e7393_, _e7453_, _e7491_);
                                            if _e7475_ {
                                                phi_31216_ = true;
                                            } else {
                                                phi_31216_ = (_e7458_ >= _e7398_);
                                            }
                                            let _e7495_ = phi_31216_;
                                            let _e7496_ = select(_e7398_, _e7458_, _e7495_);
                                            if _e7480_ {
                                                phi_31231_ = true;
                                            } else {
                                                phi_31231_ = (_e7463_ >= _e7403_);
                                            }
                                            let _e7499_ = phi_31231_;
                                            let _e7500_ = select(_e7403_, _e7463_, _e7499_);
                                            let _e7501_ = (_e7469_ != _e7469_);
                                            if _e7501_ {
                                                phi_31246_ = true;
                                            } else {
                                                phi_31246_ = (-1f >= _e7469_);
                                            }
                                            let _e7504_ = phi_31246_;
                                            let _e7505_ = select(_e7469_, -1f, _e7504_);
                                            let _e7506_ = (_e7474_ != _e7474_);
                                            if _e7506_ {
                                                phi_31261_ = true;
                                            } else {
                                                phi_31261_ = (-1f >= _e7474_);
                                            }
                                            let _e7509_ = phi_31261_;
                                            let _e7510_ = select(_e7474_, -1f, _e7509_);
                                            let _e7511_ = (_e7479_ != _e7479_);
                                            if _e7511_ {
                                                phi_31276_ = true;
                                            } else {
                                                phi_31276_ = (-1f >= _e7479_);
                                            }
                                            let _e7514_ = phi_31276_;
                                            let _e7515_ = select(_e7479_, -1f, _e7514_);
                                            let _e7516_ = (_e7484_ != _e7484_);
                                            if _e7516_ {
                                                phi_31291_ = true;
                                            } else {
                                                phi_31291_ = (-1f >= _e7484_);
                                            }
                                            let _e7519_ = phi_31291_;
                                            let _e7520_ = select(_e7484_, -1f, _e7519_);
                                            if (_e7505_ != _e7505_) {
                                                phi_31306_ = true;
                                            } else {
                                                phi_31306_ = (1f <= _e7505_);
                                            }
                                            let _e7524_ = phi_31306_;
                                            let _e7525_ = select(_e7505_, 1f, _e7524_);
                                            if (_e7510_ != _e7510_) {
                                                phi_31321_ = true;
                                            } else {
                                                phi_31321_ = (1f <= _e7510_);
                                            }
                                            let _e7529_ = phi_31321_;
                                            let _e7530_ = select(_e7510_, 1f, _e7529_);
                                            if (_e7515_ != _e7515_) {
                                                phi_31336_ = true;
                                            } else {
                                                phi_31336_ = (1f <= _e7515_);
                                            }
                                            let _e7534_ = phi_31336_;
                                            let _e7535_ = select(_e7515_, 1f, _e7534_);
                                            if (_e7520_ != _e7520_) {
                                                phi_31351_ = true;
                                            } else {
                                                phi_31351_ = (1f <= _e7520_);
                                            }
                                            let _e7539_ = phi_31351_;
                                            let _e7540_ = select(_e7520_, 1f, _e7539_);
                                            let _e7541_ = (_e7488_ != _e7488_);
                                            if _e7541_ {
                                                phi_31366_ = true;
                                            } else {
                                                phi_31366_ = (-1f >= _e7488_);
                                            }
                                            let _e7544_ = phi_31366_;
                                            let _e7545_ = select(_e7488_, -1f, _e7544_);
                                            let _e7546_ = (_e7492_ != _e7492_);
                                            if _e7546_ {
                                                phi_31381_ = true;
                                            } else {
                                                phi_31381_ = (-1f >= _e7492_);
                                            }
                                            let _e7549_ = phi_31381_;
                                            let _e7550_ = select(_e7492_, -1f, _e7549_);
                                            let _e7551_ = (_e7496_ != _e7496_);
                                            if _e7551_ {
                                                phi_31396_ = true;
                                            } else {
                                                phi_31396_ = (-1f >= _e7496_);
                                            }
                                            let _e7554_ = phi_31396_;
                                            let _e7555_ = select(_e7496_, -1f, _e7554_);
                                            let _e7556_ = (_e7500_ != _e7500_);
                                            if _e7556_ {
                                                phi_31411_ = true;
                                            } else {
                                                phi_31411_ = (-1f >= _e7500_);
                                            }
                                            let _e7559_ = phi_31411_;
                                            let _e7560_ = select(_e7500_, -1f, _e7559_);
                                            if (_e7545_ != _e7545_) {
                                                phi_31426_ = true;
                                            } else {
                                                phi_31426_ = (1f <= _e7545_);
                                            }
                                            let _e7564_ = phi_31426_;
                                            let _e7565_ = select(_e7545_, 1f, _e7564_);
                                            if (_e7550_ != _e7550_) {
                                                phi_31441_ = true;
                                            } else {
                                                phi_31441_ = (1f <= _e7550_);
                                            }
                                            let _e7569_ = phi_31441_;
                                            let _e7570_ = select(_e7550_, 1f, _e7569_);
                                            if (_e7555_ != _e7555_) {
                                                phi_31456_ = true;
                                            } else {
                                                phi_31456_ = (1f <= _e7555_);
                                            }
                                            let _e7574_ = phi_31456_;
                                            let _e7575_ = select(_e7555_, 1f, _e7574_);
                                            if (_e7560_ != _e7560_) {
                                                phi_31471_ = true;
                                            } else {
                                                phi_31471_ = (1f <= _e7560_);
                                            }
                                            let _e7579_ = phi_31471_;
                                            let _e7580_ = select(_e7560_, 1f, _e7579_);
                                            let _e7585_ = (0.5f * (_e7565_ + _e7525_));
                                            let _e7586_ = (0.5f * (_e7570_ + _e7530_));
                                            let _e7587_ = (0.5f * (_e7575_ + _e7535_));
                                            let _e7588_ = (0.5f * (_e7580_ + _e7540_));
                                            let _e7589_ = (_e7565_ - _e7525_);
                                            let _e7590_ = (_e7570_ - _e7530_);
                                            let _e7591_ = (_e7575_ - _e7535_);
                                            let _e7592_ = (_e7580_ - _e7540_);
                                            let _e7593_ = (0.5f * _e7589_);
                                            let _e7594_ = (0.5f * _e7590_);
                                            let _e7595_ = (0.5f * _e7591_);
                                            let _e7596_ = (0.5f * _e7592_);
                                            let _e7597_ = (_e7585_ * _e7585_);
                                            let _e7598_ = (_e7586_ * _e7586_);
                                            let _e7599_ = (_e7587_ * _e7587_);
                                            let _e7600_ = (_e7588_ * _e7588_);
                                            let _e7601_ = (_e7593_ * _e7593_);
                                            let _e7602_ = (_e7594_ * _e7594_);
                                            let _e7603_ = (_e7595_ * _e7595_);
                                            let _e7604_ = (_e7596_ * _e7596_);
                                            if (_e7589_ != _e7589_) {
                                                phi_31486_ = true;
                                            } else {
                                                phi_31486_ = (0f >= _e7589_);
                                            }
                                            let _e7608_ = phi_31486_;
                                            let _e7609_ = select(_e7589_, 0f, _e7608_);
                                            if (_e7590_ != _e7590_) {
                                                phi_31501_ = true;
                                            } else {
                                                phi_31501_ = (0f >= _e7590_);
                                            }
                                            let _e7613_ = phi_31501_;
                                            let _e7614_ = select(_e7590_, 0f, _e7613_);
                                            if (_e7591_ != _e7591_) {
                                                phi_31516_ = true;
                                            } else {
                                                phi_31516_ = (0f >= _e7591_);
                                            }
                                            let _e7618_ = phi_31516_;
                                            let _e7619_ = select(_e7591_, 0f, _e7618_);
                                            if (_e7592_ != _e7592_) {
                                                phi_31531_ = true;
                                            } else {
                                                phi_31531_ = (0f >= _e7592_);
                                            }
                                            let _e7623_ = phi_31531_;
                                            let _e7624_ = select(_e7592_, 0f, _e7623_);
                                            let _e7625_ = (_e7488_ - _e7469_);
                                            let _e7626_ = (_e7492_ - _e7474_);
                                            let _e7627_ = (_e7496_ - _e7479_);
                                            let _e7628_ = (_e7500_ - _e7484_);
                                            if _e7541_ {
                                                phi_31546_ = true;
                                            } else {
                                                phi_31546_ = (-1f <= _e7488_);
                                            }
                                            let _e7631_ = phi_31546_;
                                            if _e7546_ {
                                                phi_31561_ = true;
                                            } else {
                                                phi_31561_ = (-1f <= _e7492_);
                                            }
                                            let _e7635_ = phi_31561_;
                                            if _e7551_ {
                                                phi_31576_ = true;
                                            } else {
                                                phi_31576_ = (-1f <= _e7496_);
                                            }
                                            let _e7639_ = phi_31576_;
                                            if _e7556_ {
                                                phi_31591_ = true;
                                            } else {
                                                phi_31591_ = (-1f <= _e7500_);
                                            }
                                            let _e7643_ = phi_31591_;
                                            let _e7645_ = (select(_e7488_, -1f, _e7631_) - _e7469_);
                                            let _e7646_ = (select(_e7492_, -1f, _e7635_) - _e7474_);
                                            let _e7647_ = (select(_e7496_, -1f, _e7639_) - _e7479_);
                                            let _e7648_ = (select(_e7500_, -1f, _e7643_) - _e7484_);
                                            if (_e7645_ != _e7645_) {
                                                phi_31606_ = true;
                                            } else {
                                                phi_31606_ = (0f >= _e7645_);
                                            }
                                            let _e7652_ = phi_31606_;
                                            if (_e7646_ != _e7646_) {
                                                phi_31621_ = true;
                                            } else {
                                                phi_31621_ = (0f >= _e7646_);
                                            }
                                            let _e7657_ = phi_31621_;
                                            if (_e7647_ != _e7647_) {
                                                phi_31636_ = true;
                                            } else {
                                                phi_31636_ = (0f >= _e7647_);
                                            }
                                            let _e7662_ = phi_31636_;
                                            if (_e7648_ != _e7648_) {
                                                phi_31651_ = true;
                                            } else {
                                                phi_31651_ = (0f >= _e7648_);
                                            }
                                            let _e7667_ = phi_31651_;
                                            if _e7501_ {
                                                phi_31666_ = true;
                                            } else {
                                                phi_31666_ = (1f >= _e7469_);
                                            }
                                            let _e7671_ = phi_31666_;
                                            if _e7506_ {
                                                phi_31681_ = true;
                                            } else {
                                                phi_31681_ = (1f >= _e7474_);
                                            }
                                            let _e7675_ = phi_31681_;
                                            if _e7511_ {
                                                phi_31696_ = true;
                                            } else {
                                                phi_31696_ = (1f >= _e7479_);
                                            }
                                            let _e7679_ = phi_31696_;
                                            if _e7516_ {
                                                phi_31711_ = true;
                                            } else {
                                                phi_31711_ = (1f >= _e7484_);
                                            }
                                            let _e7683_ = phi_31711_;
                                            let _e7685_ = (_e7488_ - select(_e7469_, 1f, _e7671_));
                                            let _e7686_ = (_e7492_ - select(_e7474_, 1f, _e7675_));
                                            let _e7687_ = (_e7496_ - select(_e7479_, 1f, _e7679_));
                                            let _e7688_ = (_e7500_ - select(_e7484_, 1f, _e7683_));
                                            if (_e7685_ != _e7685_) {
                                                phi_31726_ = true;
                                            } else {
                                                phi_31726_ = (0f >= _e7685_);
                                            }
                                            let _e7692_ = phi_31726_;
                                            if (_e7686_ != _e7686_) {
                                                phi_31741_ = true;
                                            } else {
                                                phi_31741_ = (0f >= _e7686_);
                                            }
                                            let _e7697_ = phi_31741_;
                                            if (_e7687_ != _e7687_) {
                                                phi_31756_ = true;
                                            } else {
                                                phi_31756_ = (0f >= _e7687_);
                                            }
                                            let _e7702_ = phi_31756_;
                                            if (_e7688_ != _e7688_) {
                                                phi_31771_ = true;
                                            } else {
                                                phi_31771_ = (0f >= _e7688_);
                                            }
                                            let _e7707_ = phi_31771_;
                                            let _e7711_ = bitcast<f32>((bitcast<u32>(_e7388_) & 2147483647u));
                                            let _e7714_ = bitcast<f32>((bitcast<u32>(_e7393_) & 2147483647u));
                                            let _e7717_ = bitcast<f32>((bitcast<u32>(_e7398_) & 2147483647u));
                                            let _e7720_ = bitcast<f32>((bitcast<u32>(_e7403_) & 2147483647u));
                                            let _e7723_ = bitcast<f32>((bitcast<u32>(_e7448_) & 2147483647u));
                                            let _e7726_ = bitcast<f32>((bitcast<u32>(_e7453_) & 2147483647u));
                                            let _e7729_ = bitcast<f32>((bitcast<u32>(_e7458_) & 2147483647u));
                                            let _e7732_ = bitcast<f32>((bitcast<u32>(_e7463_) & 2147483647u));
                                            if (_e7711_ != _e7711_) {
                                                phi_31786_ = true;
                                            } else {
                                                phi_31786_ = (_e7723_ >= _e7711_);
                                            }
                                            let _e7736_ = phi_31786_;
                                            let _e7737_ = select(_e7711_, _e7723_, _e7736_);
                                            if (_e7714_ != _e7714_) {
                                                phi_31801_ = true;
                                            } else {
                                                phi_31801_ = (_e7726_ >= _e7714_);
                                            }
                                            let _e7741_ = phi_31801_;
                                            let _e7742_ = select(_e7714_, _e7726_, _e7741_);
                                            if (_e7717_ != _e7717_) {
                                                phi_31816_ = true;
                                            } else {
                                                phi_31816_ = (_e7729_ >= _e7717_);
                                            }
                                            let _e7746_ = phi_31816_;
                                            let _e7747_ = select(_e7717_, _e7729_, _e7746_);
                                            if (_e7720_ != _e7720_) {
                                                phi_31831_ = true;
                                            } else {
                                                phi_31831_ = (_e7732_ >= _e7720_);
                                            }
                                            let _e7751_ = phi_31831_;
                                            let _e7752_ = select(_e7720_, _e7732_, _e7751_);
                                            let _e7769_ = (_e7737_ <= 0f);
                                            let _e7770_ = select((_e7585_ * (1.5f - (0.5f * (_e7597_ + _e7601_)))), 0f, _e7769_);
                                            let _e7771_ = (_e7742_ <= 0f);
                                            let _e7772_ = select((_e7586_ * (1.5f - (0.5f * (_e7598_ + _e7602_)))), 0f, _e7771_);
                                            let _e7773_ = (_e7747_ <= 0f);
                                            let _e7774_ = select((_e7587_ * (1.5f - (0.5f * (_e7599_ + _e7603_)))), 0f, _e7773_);
                                            let _e7775_ = (_e7752_ <= 0f);
                                            let _e7776_ = select((_e7588_ * (1.5f - (0.5f * (_e7600_ + _e7604_)))), 0f, _e7775_);
                                            let _e7818_ = select(((_e7597_ + (0.33333334f * _e7601_)) - (((_e7597_ * _e7597_) + ((2f * _e7597_) * _e7601_)) + (0.2f * (_e7601_ * _e7601_)))), 0f, _e7769_);
                                            let _e7819_ = select(((_e7598_ + (0.33333334f * _e7602_)) - (((_e7598_ * _e7598_) + ((2f * _e7598_) * _e7602_)) + (0.2f * (_e7602_ * _e7602_)))), 0f, _e7771_);
                                            let _e7820_ = select(((_e7599_ + (0.33333334f * _e7603_)) - (((_e7599_ * _e7599_) + ((2f * _e7599_) * _e7603_)) + (0.2f * (_e7603_ * _e7603_)))), 0f, _e7773_);
                                            let _e7821_ = select(((_e7600_ + (0.33333334f * _e7604_)) - (((_e7600_ * _e7600_) + ((2f * _e7600_) * _e7604_)) + (0.2f * (_e7604_ * _e7604_)))), 0f, _e7775_);
                                            if (_e7737_ <= 1f) {
                                                if (_e7742_ <= 1f) {
                                                    let _e7825_ = (_e7747_ <= 1f);
                                                    if _e7825_ {
                                                        phi_20344_ = (_e7752_ <= 1f);
                                                    } else {
                                                        phi_20344_ = bool();
                                                    }
                                                    let _e7828_ = phi_20344_;
                                                    phi_20346_ = _e7828_;
                                                    phi_20347_ = select(true, false, _e7825_);
                                                } else {
                                                    phi_20346_ = bool();
                                                    phi_20347_ = true;
                                                }
                                                let _e7831_ = phi_20346_;
                                                let _e7833_ = phi_20347_;
                                                phi_20348_ = _e7831_;
                                                phi_20349_ = _e7833_;
                                            } else {
                                                phi_20348_ = bool();
                                                phi_20349_ = true;
                                            }
                                            let _e7835_ = phi_20348_;
                                            let _e7837_ = phi_20349_;
                                            if select(_e7835_, false, _e7837_) {
                                                phi_20727_ = type_38_(type_30_(array<f32, 4>(_e7770_, _e7772_, _e7774_, _e7776_)), type_30_(array<f32, 4>(_e7818_, _e7819_, _e7820_, _e7821_)), type_30_(_e7464_));
                                            } else {
                                                if (_e7625_ != _e7625_) {
                                                    phi_31846_ = true;
                                                } else {
                                                    phi_31846_ = (0.000001f >= _e7625_);
                                                }
                                                let _e7842_ = phi_31846_;
                                                let _e7843_ = select(_e7625_, 0.000001f, _e7842_);
                                                if (_e7626_ != _e7626_) {
                                                    phi_31861_ = true;
                                                } else {
                                                    phi_31861_ = (0.000001f >= _e7626_);
                                                }
                                                let _e7847_ = phi_31861_;
                                                let _e7848_ = select(_e7626_, 0.000001f, _e7847_);
                                                if (_e7627_ != _e7627_) {
                                                    phi_31876_ = true;
                                                } else {
                                                    phi_31876_ = (0.000001f >= _e7627_);
                                                }
                                                let _e7852_ = phi_31876_;
                                                let _e7853_ = select(_e7627_, 0.000001f, _e7852_);
                                                if (_e7628_ != _e7628_) {
                                                    phi_31891_ = true;
                                                } else {
                                                    phi_31891_ = (0.000001f >= _e7628_);
                                                }
                                                let _e7857_ = phi_31891_;
                                                let _e7858_ = select(_e7628_, 0.000001f, _e7857_);
                                                let _e7861_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7843_)));
                                                let _e7864_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7848_)));
                                                let _e7867_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7853_)));
                                                let _e7870_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7858_)));
                                                let _e7880_ = (_e7861_ + (_e7861_ * (1f - (_e7843_ * _e7861_))));
                                                let _e7882_ = (_e7864_ + (_e7864_ * (1f - (_e7848_ * _e7864_))));
                                                let _e7884_ = (_e7867_ + (_e7867_ * (1f - (_e7853_ * _e7867_))));
                                                let _e7886_ = (_e7870_ + (_e7870_ * (1f - (_e7858_ * _e7870_))));
                                                let _e7896_ = (_e7880_ + (_e7880_ * (1f - (_e7843_ * _e7880_))));
                                                let _e7898_ = (_e7882_ + (_e7882_ * (1f - (_e7848_ * _e7882_))));
                                                let _e7900_ = (_e7884_ + (_e7884_ * (1f - (_e7853_ * _e7884_))));
                                                let _e7902_ = (_e7886_ + (_e7886_ * (1f - (_e7858_ * _e7886_))));
                                                let _e7955_ = (select((select(0f, 1f, (-1f <= _e7469_)) * select(0f, 1f, (_e7488_ <= 1f))), 1f, (_e7625_ <= 0.000001f)) <= 0f);
                                                let _e7957_ = (select((select(0f, 1f, (-1f <= _e7474_)) * select(0f, 1f, (_e7492_ <= 1f))), 1f, (_e7626_ <= 0.000001f)) <= 0f);
                                                let _e7959_ = (select((select(0f, 1f, (-1f <= _e7479_)) * select(0f, 1f, (_e7496_ <= 1f))), 1f, (_e7627_ <= 0.000001f)) <= 0f);
                                                let _e7961_ = (select((select(0f, 1f, (-1f <= _e7484_)) * select(0f, 1f, (_e7500_ <= 1f))), 1f, (_e7628_ <= 0.000001f)) <= 0f);
                                                let _e7967_ = (1f <= _e7469_);
                                                let _e7969_ = (1f <= _e7474_);
                                                let _e7971_ = (1f <= _e7479_);
                                                let _e7973_ = (1f <= _e7484_);
                                                let _e7975_ = (_e7488_ <= -1f);
                                                let _e7977_ = (_e7492_ <= -1f);
                                                let _e7979_ = (_e7496_ <= -1f);
                                                let _e7981_ = (_e7500_ <= -1f);
                                                phi_20727_ = type_38_(type_30_(array<f32, 4>(select(select(select(_e7770_, (((select(_e7685_, 0f, _e7692_) - select(_e7645_, 0f, _e7652_)) + (_e7609_ * _e7770_)) * _e7896_), _e7955_), 1f, _e7967_), -1f, _e7975_), select(select(select(_e7772_, (((select(_e7686_, 0f, _e7697_) - select(_e7646_, 0f, _e7657_)) + (_e7614_ * _e7772_)) * _e7898_), _e7957_), 1f, _e7969_), -1f, _e7977_), select(select(select(_e7774_, (((select(_e7687_, 0f, _e7702_) - select(_e7647_, 0f, _e7662_)) + (_e7619_ * _e7774_)) * _e7900_), _e7959_), 1f, _e7971_), -1f, _e7979_), select(select(select(_e7776_, (((select(_e7688_, 0f, _e7707_) - select(_e7648_, 0f, _e7667_)) + (_e7624_ * _e7776_)) * _e7902_), _e7961_), 1f, _e7973_), -1f, _e7981_))), type_30_(array<f32, 4>(select(select(select(_e7818_, ((_e7609_ * _e7818_) * _e7896_), _e7955_), 0f, _e7967_), 0f, _e7975_), select(select(select(_e7819_, ((_e7614_ * _e7819_) * _e7898_), _e7957_), 0f, _e7969_), 0f, _e7977_), select(select(select(_e7820_, ((_e7619_ * _e7820_) * _e7900_), _e7959_), 0f, _e7971_), 0f, _e7979_), select(select(select(_e7821_, ((_e7624_ * _e7821_) * _e7902_), _e7961_), 0f, _e7973_), 0f, _e7981_))), type_30_(_e7464_));
                                            }
                                            let _e8002_ = phi_20727_;
                                            if (_e495_ <= 0f) {
                                                if (_e515_ <= 0f) {
                                                    let _e8006_ = (_e535_ <= 0f);
                                                    if _e8006_ {
                                                        phi_20744_ = (_e555_ <= 0f);
                                                    } else {
                                                        phi_20744_ = bool();
                                                    }
                                                    let _e8009_ = phi_20744_;
                                                    phi_20746_ = _e8009_;
                                                    phi_20747_ = select(true, false, _e8006_);
                                                } else {
                                                    phi_20746_ = bool();
                                                    phi_20747_ = true;
                                                }
                                                let _e8012_ = phi_20746_;
                                                let _e8014_ = phi_20747_;
                                                phi_20748_ = _e8012_;
                                                phi_20749_ = _e8014_;
                                            } else {
                                                phi_20748_ = bool();
                                                phi_20749_ = true;
                                            }
                                            let _e8016_ = phi_20748_;
                                            let _e8018_ = phi_20749_;
                                            if select(_e8016_, false, _e8018_) {
                                                phi_22891_ = _e8002_.member;
                                                phi_22892_ = _e7342_.member;
                                            } else {
                                                if _e6704_ {
                                                    phi_31906_ = true;
                                                } else {
                                                    phi_31906_ = (-1000000000000000000f >= _e6697_);
                                                }
                                                let _e8022_ = phi_31906_;
                                                let _e8023_ = select(_e6697_, -1000000000000000000f, _e8022_);
                                                if _e6709_ {
                                                    phi_31921_ = true;
                                                } else {
                                                    phi_31921_ = (-1000000000000000000f >= _e6699_);
                                                }
                                                let _e8026_ = phi_31921_;
                                                let _e8027_ = select(_e6699_, -1000000000000000000f, _e8026_);
                                                if _e6714_ {
                                                    phi_31936_ = true;
                                                } else {
                                                    phi_31936_ = (-1000000000000000000f >= _e6701_);
                                                }
                                                let _e8030_ = phi_31936_;
                                                let _e8031_ = select(_e6701_, -1000000000000000000f, _e8030_);
                                                if _e6719_ {
                                                    phi_31951_ = true;
                                                } else {
                                                    phi_31951_ = (-1000000000000000000f >= _e6703_);
                                                }
                                                let _e8034_ = phi_31951_;
                                                let _e8035_ = select(_e6703_, -1000000000000000000f, _e8034_);
                                                if (_e8023_ != _e8023_) {
                                                    phi_31966_ = true;
                                                } else {
                                                    phi_31966_ = (1000000000000000000f <= _e8023_);
                                                }
                                                let _e8039_ = phi_31966_;
                                                let _e8040_ = select(_e8023_, 1000000000000000000f, _e8039_);
                                                if (_e8027_ != _e8027_) {
                                                    phi_31981_ = true;
                                                } else {
                                                    phi_31981_ = (1000000000000000000f <= _e8027_);
                                                }
                                                let _e8044_ = phi_31981_;
                                                let _e8045_ = select(_e8027_, 1000000000000000000f, _e8044_);
                                                if (_e8031_ != _e8031_) {
                                                    phi_31996_ = true;
                                                } else {
                                                    phi_31996_ = (1000000000000000000f <= _e8031_);
                                                }
                                                let _e8049_ = phi_31996_;
                                                let _e8050_ = select(_e8031_, 1000000000000000000f, _e8049_);
                                                if (_e8035_ != _e8035_) {
                                                    phi_32011_ = true;
                                                } else {
                                                    phi_32011_ = (1000000000000000000f <= _e8035_);
                                                }
                                                let _e8054_ = phi_32011_;
                                                let _e8055_ = select(_e8035_, 1000000000000000000f, _e8054_);
                                                if _e6764_ {
                                                    phi_32026_ = true;
                                                } else {
                                                    phi_32026_ = (-1000000000000000000f >= _e6757_);
                                                }
                                                let _e8058_ = phi_32026_;
                                                let _e8059_ = select(_e6757_, -1000000000000000000f, _e8058_);
                                                if _e6769_ {
                                                    phi_32041_ = true;
                                                } else {
                                                    phi_32041_ = (-1000000000000000000f >= _e6759_);
                                                }
                                                let _e8062_ = phi_32041_;
                                                let _e8063_ = select(_e6759_, -1000000000000000000f, _e8062_);
                                                if _e6774_ {
                                                    phi_32056_ = true;
                                                } else {
                                                    phi_32056_ = (-1000000000000000000f >= _e6761_);
                                                }
                                                let _e8066_ = phi_32056_;
                                                let _e8067_ = select(_e6761_, -1000000000000000000f, _e8066_);
                                                if _e6779_ {
                                                    phi_32071_ = true;
                                                } else {
                                                    phi_32071_ = (-1000000000000000000f >= _e6763_);
                                                }
                                                let _e8070_ = phi_32071_;
                                                let _e8071_ = select(_e6763_, -1000000000000000000f, _e8070_);
                                                if (_e8059_ != _e8059_) {
                                                    phi_32086_ = true;
                                                } else {
                                                    phi_32086_ = (1000000000000000000f <= _e8059_);
                                                }
                                                let _e8075_ = phi_32086_;
                                                let _e8076_ = select(_e8059_, 1000000000000000000f, _e8075_);
                                                if (_e8063_ != _e8063_) {
                                                    phi_32101_ = true;
                                                } else {
                                                    phi_32101_ = (1000000000000000000f <= _e8063_);
                                                }
                                                let _e8080_ = phi_32101_;
                                                let _e8081_ = select(_e8063_, 1000000000000000000f, _e8080_);
                                                if (_e8067_ != _e8067_) {
                                                    phi_32116_ = true;
                                                } else {
                                                    phi_32116_ = (1000000000000000000f <= _e8067_);
                                                }
                                                let _e8085_ = phi_32116_;
                                                let _e8086_ = select(_e8067_, 1000000000000000000f, _e8085_);
                                                if (_e8071_ != _e8071_) {
                                                    phi_32131_ = true;
                                                } else {
                                                    phi_32131_ = (1000000000000000000f <= _e8071_);
                                                }
                                                let _e8090_ = phi_32131_;
                                                let _e8091_ = select(_e8071_, 1000000000000000000f, _e8090_);
                                                let _e8092_ = (_e8040_ != _e8040_);
                                                if _e8092_ {
                                                    phi_32146_ = true;
                                                } else {
                                                    phi_32146_ = (_e8076_ <= _e8040_);
                                                }
                                                let _e8095_ = phi_32146_;
                                                let _e8096_ = select(_e8040_, _e8076_, _e8095_);
                                                let _e8097_ = (_e8045_ != _e8045_);
                                                if _e8097_ {
                                                    phi_32161_ = true;
                                                } else {
                                                    phi_32161_ = (_e8081_ <= _e8045_);
                                                }
                                                let _e8100_ = phi_32161_;
                                                let _e8101_ = select(_e8045_, _e8081_, _e8100_);
                                                let _e8102_ = (_e8050_ != _e8050_);
                                                if _e8102_ {
                                                    phi_32176_ = true;
                                                } else {
                                                    phi_32176_ = (_e8086_ <= _e8050_);
                                                }
                                                let _e8105_ = phi_32176_;
                                                let _e8106_ = select(_e8050_, _e8086_, _e8105_);
                                                let _e8107_ = (_e8055_ != _e8055_);
                                                if _e8107_ {
                                                    phi_32191_ = true;
                                                } else {
                                                    phi_32191_ = (_e8091_ <= _e8055_);
                                                }
                                                let _e8110_ = phi_32191_;
                                                let _e8111_ = select(_e8055_, _e8091_, _e8110_);
                                                if _e8092_ {
                                                    phi_32206_ = true;
                                                } else {
                                                    phi_32206_ = (_e8076_ >= _e8040_);
                                                }
                                                let _e8114_ = phi_32206_;
                                                let _e8115_ = select(_e8040_, _e8076_, _e8114_);
                                                if _e8097_ {
                                                    phi_32221_ = true;
                                                } else {
                                                    phi_32221_ = (_e8081_ >= _e8045_);
                                                }
                                                let _e8118_ = phi_32221_;
                                                let _e8119_ = select(_e8045_, _e8081_, _e8118_);
                                                if _e8102_ {
                                                    phi_32236_ = true;
                                                } else {
                                                    phi_32236_ = (_e8086_ >= _e8050_);
                                                }
                                                let _e8122_ = phi_32236_;
                                                let _e8123_ = select(_e8050_, _e8086_, _e8122_);
                                                if _e8107_ {
                                                    phi_32251_ = true;
                                                } else {
                                                    phi_32251_ = (_e8091_ >= _e8055_);
                                                }
                                                let _e8126_ = phi_32251_;
                                                let _e8127_ = select(_e8055_, _e8091_, _e8126_);
                                                if (_e8096_ != _e8096_) {
                                                    phi_32266_ = true;
                                                } else {
                                                    phi_32266_ = (-1f >= _e8096_);
                                                }
                                                let _e8131_ = phi_32266_;
                                                let _e8132_ = select(_e8096_, -1f, _e8131_);
                                                if (_e8101_ != _e8101_) {
                                                    phi_32281_ = true;
                                                } else {
                                                    phi_32281_ = (-1f >= _e8101_);
                                                }
                                                let _e8136_ = phi_32281_;
                                                let _e8137_ = select(_e8101_, -1f, _e8136_);
                                                if (_e8106_ != _e8106_) {
                                                    phi_32296_ = true;
                                                } else {
                                                    phi_32296_ = (-1f >= _e8106_);
                                                }
                                                let _e8141_ = phi_32296_;
                                                let _e8142_ = select(_e8106_, -1f, _e8141_);
                                                if (_e8111_ != _e8111_) {
                                                    phi_32311_ = true;
                                                } else {
                                                    phi_32311_ = (-1f >= _e8111_);
                                                }
                                                let _e8146_ = phi_32311_;
                                                let _e8147_ = select(_e8111_, -1f, _e8146_);
                                                if (_e8132_ != _e8132_) {
                                                    phi_32326_ = true;
                                                } else {
                                                    phi_32326_ = (1f <= _e8132_);
                                                }
                                                let _e8151_ = phi_32326_;
                                                let _e8152_ = select(_e8132_, 1f, _e8151_);
                                                if (_e8137_ != _e8137_) {
                                                    phi_32341_ = true;
                                                } else {
                                                    phi_32341_ = (1f <= _e8137_);
                                                }
                                                let _e8156_ = phi_32341_;
                                                let _e8157_ = select(_e8137_, 1f, _e8156_);
                                                if (_e8142_ != _e8142_) {
                                                    phi_32356_ = true;
                                                } else {
                                                    phi_32356_ = (1f <= _e8142_);
                                                }
                                                let _e8161_ = phi_32356_;
                                                let _e8162_ = select(_e8142_, 1f, _e8161_);
                                                if (_e8147_ != _e8147_) {
                                                    phi_32371_ = true;
                                                } else {
                                                    phi_32371_ = (1f <= _e8147_);
                                                }
                                                let _e8166_ = phi_32371_;
                                                let _e8167_ = select(_e8147_, 1f, _e8166_);
                                                if (_e8115_ != _e8115_) {
                                                    phi_32386_ = true;
                                                } else {
                                                    phi_32386_ = (-1f >= _e8115_);
                                                }
                                                let _e8171_ = phi_32386_;
                                                let _e8172_ = select(_e8115_, -1f, _e8171_);
                                                if (_e8119_ != _e8119_) {
                                                    phi_32401_ = true;
                                                } else {
                                                    phi_32401_ = (-1f >= _e8119_);
                                                }
                                                let _e8176_ = phi_32401_;
                                                let _e8177_ = select(_e8119_, -1f, _e8176_);
                                                if (_e8123_ != _e8123_) {
                                                    phi_32416_ = true;
                                                } else {
                                                    phi_32416_ = (-1f >= _e8123_);
                                                }
                                                let _e8181_ = phi_32416_;
                                                let _e8182_ = select(_e8123_, -1f, _e8181_);
                                                if (_e8127_ != _e8127_) {
                                                    phi_32431_ = true;
                                                } else {
                                                    phi_32431_ = (-1f >= _e8127_);
                                                }
                                                let _e8186_ = phi_32431_;
                                                let _e8187_ = select(_e8127_, -1f, _e8186_);
                                                if (_e8172_ != _e8172_) {
                                                    phi_32446_ = true;
                                                } else {
                                                    phi_32446_ = (1f <= _e8172_);
                                                }
                                                let _e8191_ = phi_32446_;
                                                let _e8192_ = select(_e8172_, 1f, _e8191_);
                                                if (_e8177_ != _e8177_) {
                                                    phi_32461_ = true;
                                                } else {
                                                    phi_32461_ = (1f <= _e8177_);
                                                }
                                                let _e8196_ = phi_32461_;
                                                let _e8197_ = select(_e8177_, 1f, _e8196_);
                                                if (_e8182_ != _e8182_) {
                                                    phi_32476_ = true;
                                                } else {
                                                    phi_32476_ = (1f <= _e8182_);
                                                }
                                                let _e8201_ = phi_32476_;
                                                let _e8202_ = select(_e8182_, 1f, _e8201_);
                                                if (_e8187_ != _e8187_) {
                                                    phi_32491_ = true;
                                                } else {
                                                    phi_32491_ = (1f <= _e8187_);
                                                }
                                                let _e8206_ = phi_32491_;
                                                let _e8207_ = select(_e8187_, 1f, _e8206_);
                                                let _e8212_ = (0.5f * (_e8192_ + _e8152_));
                                                let _e8213_ = (0.5f * (_e8197_ + _e8157_));
                                                let _e8214_ = (0.5f * (_e8202_ + _e8162_));
                                                let _e8215_ = (0.5f * (_e8207_ + _e8167_));
                                                let _e8216_ = (_e8192_ - _e8152_);
                                                let _e8217_ = (_e8197_ - _e8157_);
                                                let _e8218_ = (_e8202_ - _e8162_);
                                                let _e8219_ = (_e8207_ - _e8167_);
                                                let _e8220_ = (0.5f * _e8216_);
                                                let _e8221_ = (0.5f * _e8217_);
                                                let _e8222_ = (0.5f * _e8218_);
                                                let _e8223_ = (0.5f * _e8219_);
                                                let _e8224_ = (_e8212_ * _e8212_);
                                                let _e8225_ = (_e8213_ * _e8213_);
                                                let _e8226_ = (_e8214_ * _e8214_);
                                                let _e8227_ = (_e8215_ * _e8215_);
                                                let _e8228_ = (_e8220_ * _e8220_);
                                                let _e8229_ = (_e8221_ * _e8221_);
                                                let _e8230_ = (_e8222_ * _e8222_);
                                                let _e8231_ = (_e8223_ * _e8223_);
                                                if (_e8216_ != _e8216_) {
                                                    phi_32506_ = true;
                                                } else {
                                                    phi_32506_ = (0f >= _e8216_);
                                                }
                                                let _e8235_ = phi_32506_;
                                                let _e8236_ = select(_e8216_, 0f, _e8235_);
                                                if (_e8217_ != _e8217_) {
                                                    phi_32521_ = true;
                                                } else {
                                                    phi_32521_ = (0f >= _e8217_);
                                                }
                                                let _e8240_ = phi_32521_;
                                                let _e8241_ = select(_e8217_, 0f, _e8240_);
                                                if (_e8218_ != _e8218_) {
                                                    phi_32536_ = true;
                                                } else {
                                                    phi_32536_ = (0f >= _e8218_);
                                                }
                                                let _e8245_ = phi_32536_;
                                                let _e8246_ = select(_e8218_, 0f, _e8245_);
                                                if (_e8219_ != _e8219_) {
                                                    phi_32551_ = true;
                                                } else {
                                                    phi_32551_ = (0f >= _e8219_);
                                                }
                                                let _e8250_ = phi_32551_;
                                                let _e8251_ = select(_e8219_, 0f, _e8250_);
                                                let _e8252_ = (_e8115_ - _e8096_);
                                                let _e8253_ = (_e8119_ - _e8101_);
                                                let _e8254_ = (_e8123_ - _e8106_);
                                                let _e8255_ = (_e8127_ - _e8111_);
                                                let _e8258_ = bitcast<f32>((bitcast<u32>(_e8040_) & 2147483647u));
                                                let _e8261_ = bitcast<f32>((bitcast<u32>(_e8045_) & 2147483647u));
                                                let _e8264_ = bitcast<f32>((bitcast<u32>(_e8050_) & 2147483647u));
                                                let _e8267_ = bitcast<f32>((bitcast<u32>(_e8055_) & 2147483647u));
                                                let _e8270_ = bitcast<f32>((bitcast<u32>(_e8076_) & 2147483647u));
                                                let _e8273_ = bitcast<f32>((bitcast<u32>(_e8081_) & 2147483647u));
                                                let _e8276_ = bitcast<f32>((bitcast<u32>(_e8086_) & 2147483647u));
                                                let _e8279_ = bitcast<f32>((bitcast<u32>(_e8091_) & 2147483647u));
                                                if (_e8258_ != _e8258_) {
                                                    phi_32806_ = true;
                                                } else {
                                                    phi_32806_ = (_e8270_ >= _e8258_);
                                                }
                                                let _e8283_ = phi_32806_;
                                                let _e8284_ = select(_e8258_, _e8270_, _e8283_);
                                                if (_e8261_ != _e8261_) {
                                                    phi_32821_ = true;
                                                } else {
                                                    phi_32821_ = (_e8273_ >= _e8261_);
                                                }
                                                let _e8288_ = phi_32821_;
                                                let _e8289_ = select(_e8261_, _e8273_, _e8288_);
                                                if (_e8264_ != _e8264_) {
                                                    phi_32836_ = true;
                                                } else {
                                                    phi_32836_ = (_e8276_ >= _e8264_);
                                                }
                                                let _e8293_ = phi_32836_;
                                                let _e8294_ = select(_e8264_, _e8276_, _e8293_);
                                                if (_e8267_ != _e8267_) {
                                                    phi_32851_ = true;
                                                } else {
                                                    phi_32851_ = (_e8279_ >= _e8267_);
                                                }
                                                let _e8298_ = phi_32851_;
                                                let _e8299_ = select(_e8267_, _e8279_, _e8298_);
                                                let _e8304_ = (1f - (_e8224_ + _e8228_));
                                                let _e8305_ = (1f - (_e8225_ + _e8229_));
                                                let _e8306_ = (1f - (_e8226_ + _e8230_));
                                                let _e8307_ = (1f - (_e8227_ + _e8231_));
                                                let _e8316_ = (4f * _e8224_);
                                                let _e8317_ = (4f * _e8225_);
                                                let _e8318_ = (4f * _e8226_);
                                                let _e8319_ = (4f * _e8227_);
                                                let _e8332_ = (_e8284_ <= 0f);
                                                let _e8333_ = select(((_e8212_ * _e8304_) * ((_e8304_ * _e8304_) + (_e8316_ * _e8228_))), 0f, _e8332_);
                                                let _e8334_ = (_e8289_ <= 0f);
                                                let _e8335_ = select(((_e8213_ * _e8305_) * ((_e8305_ * _e8305_) + (_e8317_ * _e8229_))), 0f, _e8334_);
                                                let _e8336_ = (_e8294_ <= 0f);
                                                let _e8337_ = select(((_e8214_ * _e8306_) * ((_e8306_ * _e8306_) + (_e8318_ * _e8230_))), 0f, _e8336_);
                                                let _e8338_ = (_e8299_ <= 0f);
                                                let _e8339_ = select(((_e8215_ * _e8307_) * ((_e8307_ * _e8307_) + (_e8319_ * _e8231_))), 0f, _e8338_);
                                                let _e8341_ = (_e8224_ * _e8224_);
                                                let _e8342_ = (_e8225_ * _e8225_);
                                                let _e8343_ = (_e8226_ * _e8226_);
                                                let _e8344_ = (_e8227_ * _e8227_);
                                                let _e8345_ = (_e8228_ * _e8228_);
                                                let _e8346_ = (_e8229_ * _e8229_);
                                                let _e8347_ = (_e8230_ * _e8230_);
                                                let _e8348_ = (_e8231_ * _e8231_);
                                                let _e8349_ = (_e8341_ * _e8224_);
                                                let _e8350_ = (_e8342_ * _e8225_);
                                                let _e8351_ = (_e8343_ * _e8226_);
                                                let _e8352_ = (_e8344_ * _e8227_);
                                                let _e8353_ = (_e8345_ * _e8228_);
                                                let _e8354_ = (_e8346_ * _e8229_);
                                                let _e8355_ = (_e8347_ * _e8230_);
                                                let _e8356_ = (_e8348_ * _e8231_);
                                                let _e8481_ = select((((_e8224_ + (0.33333334f * _e8228_)) - (((((_e8341_ * _e8341_) + ((9.333333f * _e8349_) * _e8228_)) + ((14f * _e8341_) * _e8345_)) + (_e8316_ * _e8353_)) + (0.11111111f * (_e8345_ * _e8345_)))) - (3f * (((_e8341_ + ((2f * _e8224_) * _e8228_)) + (0.2f * _e8345_)) - (((_e8349_ + ((5f * _e8341_) * _e8228_)) + ((3f * _e8224_) * _e8345_)) + (0.14285715f * _e8353_))))), 0f, _e8332_);
                                                let _e8482_ = select((((_e8225_ + (0.33333334f * _e8229_)) - (((((_e8342_ * _e8342_) + ((9.333333f * _e8350_) * _e8229_)) + ((14f * _e8342_) * _e8346_)) + (_e8317_ * _e8354_)) + (0.11111111f * (_e8346_ * _e8346_)))) - (3f * (((_e8342_ + ((2f * _e8225_) * _e8229_)) + (0.2f * _e8346_)) - (((_e8350_ + ((5f * _e8342_) * _e8229_)) + ((3f * _e8225_) * _e8346_)) + (0.14285715f * _e8354_))))), 0f, _e8334_);
                                                let _e8483_ = select((((_e8226_ + (0.33333334f * _e8230_)) - (((((_e8343_ * _e8343_) + ((9.333333f * _e8351_) * _e8230_)) + ((14f * _e8343_) * _e8347_)) + (_e8318_ * _e8355_)) + (0.11111111f * (_e8347_ * _e8347_)))) - (3f * (((_e8343_ + ((2f * _e8226_) * _e8230_)) + (0.2f * _e8347_)) - (((_e8351_ + ((5f * _e8343_) * _e8230_)) + ((3f * _e8226_) * _e8347_)) + (0.14285715f * _e8355_))))), 0f, _e8336_);
                                                let _e8484_ = select((((_e8227_ + (0.33333334f * _e8231_)) - (((((_e8344_ * _e8344_) + ((9.333333f * _e8352_) * _e8231_)) + ((14f * _e8344_) * _e8348_)) + (_e8319_ * _e8356_)) + (0.11111111f * (_e8348_ * _e8348_)))) - (3f * (((_e8344_ + ((2f * _e8227_) * _e8231_)) + (0.2f * _e8348_)) - (((_e8352_ + ((5f * _e8344_) * _e8231_)) + ((3f * _e8227_) * _e8348_)) + (0.14285715f * _e8356_))))), 0f, _e8338_);
                                                if (_e8284_ <= 1f) {
                                                    if (_e8289_ <= 1f) {
                                                        let _e8488_ = (_e8294_ <= 1f);
                                                        if _e8488_ {
                                                            phi_21359_ = (_e8299_ <= 1f);
                                                        } else {
                                                            phi_21359_ = bool();
                                                        }
                                                        let _e8491_ = phi_21359_;
                                                        phi_21361_ = _e8491_;
                                                        phi_21362_ = select(true, false, _e8488_);
                                                    } else {
                                                        phi_21361_ = bool();
                                                        phi_21362_ = true;
                                                    }
                                                    let _e8494_ = phi_21361_;
                                                    let _e8496_ = phi_21362_;
                                                    phi_21363_ = _e8494_;
                                                    phi_21364_ = _e8496_;
                                                } else {
                                                    phi_21363_ = bool();
                                                    phi_21364_ = true;
                                                }
                                                let _e8498_ = phi_21363_;
                                                let _e8500_ = phi_21364_;
                                                if select(_e8498_, false, _e8500_) {
                                                    phi_21796_ = type_31_(type_30_(array<f32, 4>(_e8333_, _e8335_, _e8337_, _e8339_)), type_30_(array<f32, 4>(_e8481_, _e8482_, _e8483_, _e8484_)));
                                                } else {
                                                    if (_e8252_ != _e8252_) {
                                                        phi_32866_ = true;
                                                    } else {
                                                        phi_32866_ = (0.000001f >= _e8252_);
                                                    }
                                                    let _e8505_ = phi_32866_;
                                                    let _e8506_ = select(_e8252_, 0.000001f, _e8505_);
                                                    if (_e8253_ != _e8253_) {
                                                        phi_32881_ = true;
                                                    } else {
                                                        phi_32881_ = (0.000001f >= _e8253_);
                                                    }
                                                    let _e8510_ = phi_32881_;
                                                    let _e8511_ = select(_e8253_, 0.000001f, _e8510_);
                                                    if (_e8254_ != _e8254_) {
                                                        phi_32896_ = true;
                                                    } else {
                                                        phi_32896_ = (0.000001f >= _e8254_);
                                                    }
                                                    let _e8515_ = phi_32896_;
                                                    let _e8516_ = select(_e8254_, 0.000001f, _e8515_);
                                                    if (_e8255_ != _e8255_) {
                                                        phi_32911_ = true;
                                                    } else {
                                                        phi_32911_ = (0.000001f >= _e8255_);
                                                    }
                                                    let _e8520_ = phi_32911_;
                                                    let _e8521_ = select(_e8255_, 0.000001f, _e8520_);
                                                    let _e8524_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8506_)));
                                                    let _e8527_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8511_)));
                                                    let _e8530_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8516_)));
                                                    let _e8533_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8521_)));
                                                    let _e8543_ = (_e8524_ + (_e8524_ * (1f - (_e8506_ * _e8524_))));
                                                    let _e8545_ = (_e8527_ + (_e8527_ * (1f - (_e8511_ * _e8527_))));
                                                    let _e8547_ = (_e8530_ + (_e8530_ * (1f - (_e8516_ * _e8530_))));
                                                    let _e8549_ = (_e8533_ + (_e8533_ * (1f - (_e8521_ * _e8533_))));
                                                    let _e8559_ = (_e8543_ + (_e8543_ * (1f - (_e8506_ * _e8543_))));
                                                    let _e8561_ = (_e8545_ + (_e8545_ * (1f - (_e8511_ * _e8545_))));
                                                    let _e8563_ = (_e8547_ + (_e8547_ * (1f - (_e8516_ * _e8547_))));
                                                    let _e8565_ = (_e8549_ + (_e8549_ * (1f - (_e8521_ * _e8549_))));
                                                    let _e8602_ = (select((select(0f, 1f, (-1f <= _e8096_)) * select(0f, 1f, (_e8115_ <= 1f))), 1f, (_e8252_ <= 0.000001f)) <= 0f);
                                                    let _e8604_ = (select((select(0f, 1f, (-1f <= _e8101_)) * select(0f, 1f, (_e8119_ <= 1f))), 1f, (_e8253_ <= 0.000001f)) <= 0f);
                                                    let _e8606_ = (select((select(0f, 1f, (-1f <= _e8106_)) * select(0f, 1f, (_e8123_ <= 1f))), 1f, (_e8254_ <= 0.000001f)) <= 0f);
                                                    let _e8608_ = (select((select(0f, 1f, (-1f <= _e8111_)) * select(0f, 1f, (_e8127_ <= 1f))), 1f, (_e8255_ <= 0.000001f)) <= 0f);
                                                    let _e8610_ = (1f <= _e8096_);
                                                    let _e8612_ = (1f <= _e8101_);
                                                    let _e8614_ = (1f <= _e8106_);
                                                    let _e8616_ = (1f <= _e8111_);
                                                    let _e8618_ = (_e8115_ <= -1f);
                                                    let _e8620_ = (_e8119_ <= -1f);
                                                    let _e8622_ = (_e8123_ <= -1f);
                                                    let _e8624_ = (_e8127_ <= -1f);
                                                    phi_21796_ = type_31_(type_30_(array<f32, 4>(select(select(select(_e8333_, ((_e8236_ * _e8333_) * _e8559_), _e8602_), 0f, _e8610_), 0f, _e8618_), select(select(select(_e8335_, ((_e8241_ * _e8335_) * _e8561_), _e8604_), 0f, _e8612_), 0f, _e8620_), select(select(select(_e8337_, ((_e8246_ * _e8337_) * _e8563_), _e8606_), 0f, _e8614_), 0f, _e8622_), select(select(select(_e8339_, ((_e8251_ * _e8339_) * _e8565_), _e8608_), 0f, _e8616_), 0f, _e8624_))), type_30_(array<f32, 4>(select(select(select(_e8481_, ((_e8236_ * _e8481_) * _e8559_), _e8602_), 0f, _e8610_), 0f, _e8618_), select(select(select(_e8482_, ((_e8241_ * _e8482_) * _e8561_), _e8604_), 0f, _e8612_), 0f, _e8620_), select(select(select(_e8483_, ((_e8246_ * _e8483_) * _e8563_), _e8606_), 0f, _e8614_), 0f, _e8622_), select(select(select(_e8484_, ((_e8251_ * _e8484_) * _e8565_), _e8608_), 0f, _e8616_), 0f, _e8624_))));
                                                }
                                                let _e8655_ = phi_21796_;
                                                if _e7364_ {
                                                    phi_32926_ = true;
                                                } else {
                                                    phi_32926_ = (-1000000000000000000f >= _e7357_);
                                                }
                                                let _e8658_ = phi_32926_;
                                                let _e8659_ = select(_e7357_, -1000000000000000000f, _e8658_);
                                                if _e7369_ {
                                                    phi_32941_ = true;
                                                } else {
                                                    phi_32941_ = (-1000000000000000000f >= _e7359_);
                                                }
                                                let _e8662_ = phi_32941_;
                                                let _e8663_ = select(_e7359_, -1000000000000000000f, _e8662_);
                                                if _e7374_ {
                                                    phi_32956_ = true;
                                                } else {
                                                    phi_32956_ = (-1000000000000000000f >= _e7361_);
                                                }
                                                let _e8666_ = phi_32956_;
                                                let _e8667_ = select(_e7361_, -1000000000000000000f, _e8666_);
                                                if _e7379_ {
                                                    phi_32971_ = true;
                                                } else {
                                                    phi_32971_ = (-1000000000000000000f >= _e7363_);
                                                }
                                                let _e8670_ = phi_32971_;
                                                let _e8671_ = select(_e7363_, -1000000000000000000f, _e8670_);
                                                if (_e8659_ != _e8659_) {
                                                    phi_32986_ = true;
                                                } else {
                                                    phi_32986_ = (1000000000000000000f <= _e8659_);
                                                }
                                                let _e8675_ = phi_32986_;
                                                let _e8676_ = select(_e8659_, 1000000000000000000f, _e8675_);
                                                if (_e8663_ != _e8663_) {
                                                    phi_33001_ = true;
                                                } else {
                                                    phi_33001_ = (1000000000000000000f <= _e8663_);
                                                }
                                                let _e8680_ = phi_33001_;
                                                let _e8681_ = select(_e8663_, 1000000000000000000f, _e8680_);
                                                if (_e8667_ != _e8667_) {
                                                    phi_33016_ = true;
                                                } else {
                                                    phi_33016_ = (1000000000000000000f <= _e8667_);
                                                }
                                                let _e8685_ = phi_33016_;
                                                let _e8686_ = select(_e8667_, 1000000000000000000f, _e8685_);
                                                if (_e8671_ != _e8671_) {
                                                    phi_33031_ = true;
                                                } else {
                                                    phi_33031_ = (1000000000000000000f <= _e8671_);
                                                }
                                                let _e8690_ = phi_33031_;
                                                let _e8691_ = select(_e8671_, 1000000000000000000f, _e8690_);
                                                if _e7424_ {
                                                    phi_33046_ = true;
                                                } else {
                                                    phi_33046_ = (-1000000000000000000f >= _e7417_);
                                                }
                                                let _e8694_ = phi_33046_;
                                                let _e8695_ = select(_e7417_, -1000000000000000000f, _e8694_);
                                                if _e7429_ {
                                                    phi_33061_ = true;
                                                } else {
                                                    phi_33061_ = (-1000000000000000000f >= _e7419_);
                                                }
                                                let _e8698_ = phi_33061_;
                                                let _e8699_ = select(_e7419_, -1000000000000000000f, _e8698_);
                                                if _e7434_ {
                                                    phi_33076_ = true;
                                                } else {
                                                    phi_33076_ = (-1000000000000000000f >= _e7421_);
                                                }
                                                let _e8702_ = phi_33076_;
                                                let _e8703_ = select(_e7421_, -1000000000000000000f, _e8702_);
                                                if _e7439_ {
                                                    phi_33091_ = true;
                                                } else {
                                                    phi_33091_ = (-1000000000000000000f >= _e7423_);
                                                }
                                                let _e8706_ = phi_33091_;
                                                let _e8707_ = select(_e7423_, -1000000000000000000f, _e8706_);
                                                if (_e8695_ != _e8695_) {
                                                    phi_33106_ = true;
                                                } else {
                                                    phi_33106_ = (1000000000000000000f <= _e8695_);
                                                }
                                                let _e8711_ = phi_33106_;
                                                let _e8712_ = select(_e8695_, 1000000000000000000f, _e8711_);
                                                if (_e8699_ != _e8699_) {
                                                    phi_33121_ = true;
                                                } else {
                                                    phi_33121_ = (1000000000000000000f <= _e8699_);
                                                }
                                                let _e8716_ = phi_33121_;
                                                let _e8717_ = select(_e8699_, 1000000000000000000f, _e8716_);
                                                if (_e8703_ != _e8703_) {
                                                    phi_33136_ = true;
                                                } else {
                                                    phi_33136_ = (1000000000000000000f <= _e8703_);
                                                }
                                                let _e8721_ = phi_33136_;
                                                let _e8722_ = select(_e8703_, 1000000000000000000f, _e8721_);
                                                if (_e8707_ != _e8707_) {
                                                    phi_33151_ = true;
                                                } else {
                                                    phi_33151_ = (1000000000000000000f <= _e8707_);
                                                }
                                                let _e8726_ = phi_33151_;
                                                let _e8727_ = select(_e8707_, 1000000000000000000f, _e8726_);
                                                let _e8728_ = (_e8676_ != _e8676_);
                                                if _e8728_ {
                                                    phi_33166_ = true;
                                                } else {
                                                    phi_33166_ = (_e8712_ <= _e8676_);
                                                }
                                                let _e8731_ = phi_33166_;
                                                let _e8732_ = select(_e8676_, _e8712_, _e8731_);
                                                let _e8733_ = (_e8681_ != _e8681_);
                                                if _e8733_ {
                                                    phi_33181_ = true;
                                                } else {
                                                    phi_33181_ = (_e8717_ <= _e8681_);
                                                }
                                                let _e8736_ = phi_33181_;
                                                let _e8737_ = select(_e8681_, _e8717_, _e8736_);
                                                let _e8738_ = (_e8686_ != _e8686_);
                                                if _e8738_ {
                                                    phi_33196_ = true;
                                                } else {
                                                    phi_33196_ = (_e8722_ <= _e8686_);
                                                }
                                                let _e8741_ = phi_33196_;
                                                let _e8742_ = select(_e8686_, _e8722_, _e8741_);
                                                let _e8743_ = (_e8691_ != _e8691_);
                                                if _e8743_ {
                                                    phi_33211_ = true;
                                                } else {
                                                    phi_33211_ = (_e8727_ <= _e8691_);
                                                }
                                                let _e8746_ = phi_33211_;
                                                let _e8747_ = select(_e8691_, _e8727_, _e8746_);
                                                if _e8728_ {
                                                    phi_33226_ = true;
                                                } else {
                                                    phi_33226_ = (_e8712_ >= _e8676_);
                                                }
                                                let _e8750_ = phi_33226_;
                                                let _e8751_ = select(_e8676_, _e8712_, _e8750_);
                                                if _e8733_ {
                                                    phi_33241_ = true;
                                                } else {
                                                    phi_33241_ = (_e8717_ >= _e8681_);
                                                }
                                                let _e8754_ = phi_33241_;
                                                let _e8755_ = select(_e8681_, _e8717_, _e8754_);
                                                if _e8738_ {
                                                    phi_33256_ = true;
                                                } else {
                                                    phi_33256_ = (_e8722_ >= _e8686_);
                                                }
                                                let _e8758_ = phi_33256_;
                                                let _e8759_ = select(_e8686_, _e8722_, _e8758_);
                                                if _e8743_ {
                                                    phi_33271_ = true;
                                                } else {
                                                    phi_33271_ = (_e8727_ >= _e8691_);
                                                }
                                                let _e8762_ = phi_33271_;
                                                let _e8763_ = select(_e8691_, _e8727_, _e8762_);
                                                if (_e8732_ != _e8732_) {
                                                    phi_33286_ = true;
                                                } else {
                                                    phi_33286_ = (-1f >= _e8732_);
                                                }
                                                let _e8767_ = phi_33286_;
                                                let _e8768_ = select(_e8732_, -1f, _e8767_);
                                                if (_e8737_ != _e8737_) {
                                                    phi_33301_ = true;
                                                } else {
                                                    phi_33301_ = (-1f >= _e8737_);
                                                }
                                                let _e8772_ = phi_33301_;
                                                let _e8773_ = select(_e8737_, -1f, _e8772_);
                                                if (_e8742_ != _e8742_) {
                                                    phi_33316_ = true;
                                                } else {
                                                    phi_33316_ = (-1f >= _e8742_);
                                                }
                                                let _e8777_ = phi_33316_;
                                                let _e8778_ = select(_e8742_, -1f, _e8777_);
                                                if (_e8747_ != _e8747_) {
                                                    phi_33331_ = true;
                                                } else {
                                                    phi_33331_ = (-1f >= _e8747_);
                                                }
                                                let _e8782_ = phi_33331_;
                                                let _e8783_ = select(_e8747_, -1f, _e8782_);
                                                if (_e8768_ != _e8768_) {
                                                    phi_33346_ = true;
                                                } else {
                                                    phi_33346_ = (1f <= _e8768_);
                                                }
                                                let _e8787_ = phi_33346_;
                                                let _e8788_ = select(_e8768_, 1f, _e8787_);
                                                if (_e8773_ != _e8773_) {
                                                    phi_33361_ = true;
                                                } else {
                                                    phi_33361_ = (1f <= _e8773_);
                                                }
                                                let _e8792_ = phi_33361_;
                                                let _e8793_ = select(_e8773_, 1f, _e8792_);
                                                if (_e8778_ != _e8778_) {
                                                    phi_33376_ = true;
                                                } else {
                                                    phi_33376_ = (1f <= _e8778_);
                                                }
                                                let _e8797_ = phi_33376_;
                                                let _e8798_ = select(_e8778_, 1f, _e8797_);
                                                if (_e8783_ != _e8783_) {
                                                    phi_33391_ = true;
                                                } else {
                                                    phi_33391_ = (1f <= _e8783_);
                                                }
                                                let _e8802_ = phi_33391_;
                                                let _e8803_ = select(_e8783_, 1f, _e8802_);
                                                if (_e8751_ != _e8751_) {
                                                    phi_33406_ = true;
                                                } else {
                                                    phi_33406_ = (-1f >= _e8751_);
                                                }
                                                let _e8807_ = phi_33406_;
                                                let _e8808_ = select(_e8751_, -1f, _e8807_);
                                                if (_e8755_ != _e8755_) {
                                                    phi_33421_ = true;
                                                } else {
                                                    phi_33421_ = (-1f >= _e8755_);
                                                }
                                                let _e8812_ = phi_33421_;
                                                let _e8813_ = select(_e8755_, -1f, _e8812_);
                                                if (_e8759_ != _e8759_) {
                                                    phi_33436_ = true;
                                                } else {
                                                    phi_33436_ = (-1f >= _e8759_);
                                                }
                                                let _e8817_ = phi_33436_;
                                                let _e8818_ = select(_e8759_, -1f, _e8817_);
                                                if (_e8763_ != _e8763_) {
                                                    phi_33451_ = true;
                                                } else {
                                                    phi_33451_ = (-1f >= _e8763_);
                                                }
                                                let _e8822_ = phi_33451_;
                                                let _e8823_ = select(_e8763_, -1f, _e8822_);
                                                if (_e8808_ != _e8808_) {
                                                    phi_33466_ = true;
                                                } else {
                                                    phi_33466_ = (1f <= _e8808_);
                                                }
                                                let _e8827_ = phi_33466_;
                                                let _e8828_ = select(_e8808_, 1f, _e8827_);
                                                if (_e8813_ != _e8813_) {
                                                    phi_33481_ = true;
                                                } else {
                                                    phi_33481_ = (1f <= _e8813_);
                                                }
                                                let _e8832_ = phi_33481_;
                                                let _e8833_ = select(_e8813_, 1f, _e8832_);
                                                if (_e8818_ != _e8818_) {
                                                    phi_33496_ = true;
                                                } else {
                                                    phi_33496_ = (1f <= _e8818_);
                                                }
                                                let _e8837_ = phi_33496_;
                                                let _e8838_ = select(_e8818_, 1f, _e8837_);
                                                if (_e8823_ != _e8823_) {
                                                    phi_33511_ = true;
                                                } else {
                                                    phi_33511_ = (1f <= _e8823_);
                                                }
                                                let _e8842_ = phi_33511_;
                                                let _e8843_ = select(_e8823_, 1f, _e8842_);
                                                let _e8848_ = (0.5f * (_e8828_ + _e8788_));
                                                let _e8849_ = (0.5f * (_e8833_ + _e8793_));
                                                let _e8850_ = (0.5f * (_e8838_ + _e8798_));
                                                let _e8851_ = (0.5f * (_e8843_ + _e8803_));
                                                let _e8852_ = (_e8828_ - _e8788_);
                                                let _e8853_ = (_e8833_ - _e8793_);
                                                let _e8854_ = (_e8838_ - _e8798_);
                                                let _e8855_ = (_e8843_ - _e8803_);
                                                let _e8856_ = (0.5f * _e8852_);
                                                let _e8857_ = (0.5f * _e8853_);
                                                let _e8858_ = (0.5f * _e8854_);
                                                let _e8859_ = (0.5f * _e8855_);
                                                let _e8860_ = (_e8848_ * _e8848_);
                                                let _e8861_ = (_e8849_ * _e8849_);
                                                let _e8862_ = (_e8850_ * _e8850_);
                                                let _e8863_ = (_e8851_ * _e8851_);
                                                let _e8864_ = (_e8856_ * _e8856_);
                                                let _e8865_ = (_e8857_ * _e8857_);
                                                let _e8866_ = (_e8858_ * _e8858_);
                                                let _e8867_ = (_e8859_ * _e8859_);
                                                if (_e8852_ != _e8852_) {
                                                    phi_33526_ = true;
                                                } else {
                                                    phi_33526_ = (0f >= _e8852_);
                                                }
                                                let _e8871_ = phi_33526_;
                                                let _e8872_ = select(_e8852_, 0f, _e8871_);
                                                if (_e8853_ != _e8853_) {
                                                    phi_33541_ = true;
                                                } else {
                                                    phi_33541_ = (0f >= _e8853_);
                                                }
                                                let _e8876_ = phi_33541_;
                                                let _e8877_ = select(_e8853_, 0f, _e8876_);
                                                if (_e8854_ != _e8854_) {
                                                    phi_33556_ = true;
                                                } else {
                                                    phi_33556_ = (0f >= _e8854_);
                                                }
                                                let _e8881_ = phi_33556_;
                                                let _e8882_ = select(_e8854_, 0f, _e8881_);
                                                if (_e8855_ != _e8855_) {
                                                    phi_33571_ = true;
                                                } else {
                                                    phi_33571_ = (0f >= _e8855_);
                                                }
                                                let _e8886_ = phi_33571_;
                                                let _e8887_ = select(_e8855_, 0f, _e8886_);
                                                let _e8888_ = (_e8751_ - _e8732_);
                                                let _e8889_ = (_e8755_ - _e8737_);
                                                let _e8890_ = (_e8759_ - _e8742_);
                                                let _e8891_ = (_e8763_ - _e8747_);
                                                let _e8894_ = bitcast<f32>((bitcast<u32>(_e8676_) & 2147483647u));
                                                let _e8897_ = bitcast<f32>((bitcast<u32>(_e8681_) & 2147483647u));
                                                let _e8900_ = bitcast<f32>((bitcast<u32>(_e8686_) & 2147483647u));
                                                let _e8903_ = bitcast<f32>((bitcast<u32>(_e8691_) & 2147483647u));
                                                let _e8906_ = bitcast<f32>((bitcast<u32>(_e8712_) & 2147483647u));
                                                let _e8909_ = bitcast<f32>((bitcast<u32>(_e8717_) & 2147483647u));
                                                let _e8912_ = bitcast<f32>((bitcast<u32>(_e8722_) & 2147483647u));
                                                let _e8915_ = bitcast<f32>((bitcast<u32>(_e8727_) & 2147483647u));
                                                if (_e8894_ != _e8894_) {
                                                    phi_33826_ = true;
                                                } else {
                                                    phi_33826_ = (_e8906_ >= _e8894_);
                                                }
                                                let _e8919_ = phi_33826_;
                                                let _e8920_ = select(_e8894_, _e8906_, _e8919_);
                                                if (_e8897_ != _e8897_) {
                                                    phi_33841_ = true;
                                                } else {
                                                    phi_33841_ = (_e8909_ >= _e8897_);
                                                }
                                                let _e8924_ = phi_33841_;
                                                let _e8925_ = select(_e8897_, _e8909_, _e8924_);
                                                if (_e8900_ != _e8900_) {
                                                    phi_33856_ = true;
                                                } else {
                                                    phi_33856_ = (_e8912_ >= _e8900_);
                                                }
                                                let _e8929_ = phi_33856_;
                                                let _e8930_ = select(_e8900_, _e8912_, _e8929_);
                                                if (_e8903_ != _e8903_) {
                                                    phi_33871_ = true;
                                                } else {
                                                    phi_33871_ = (_e8915_ >= _e8903_);
                                                }
                                                let _e8934_ = phi_33871_;
                                                let _e8935_ = select(_e8903_, _e8915_, _e8934_);
                                                let _e8940_ = (1f - (_e8860_ + _e8864_));
                                                let _e8941_ = (1f - (_e8861_ + _e8865_));
                                                let _e8942_ = (1f - (_e8862_ + _e8866_));
                                                let _e8943_ = (1f - (_e8863_ + _e8867_));
                                                let _e8952_ = (4f * _e8860_);
                                                let _e8953_ = (4f * _e8861_);
                                                let _e8954_ = (4f * _e8862_);
                                                let _e8955_ = (4f * _e8863_);
                                                let _e8968_ = (_e8920_ <= 0f);
                                                let _e8969_ = select(((_e8848_ * _e8940_) * ((_e8940_ * _e8940_) + (_e8952_ * _e8864_))), 0f, _e8968_);
                                                let _e8970_ = (_e8925_ <= 0f);
                                                let _e8971_ = select(((_e8849_ * _e8941_) * ((_e8941_ * _e8941_) + (_e8953_ * _e8865_))), 0f, _e8970_);
                                                let _e8972_ = (_e8930_ <= 0f);
                                                let _e8973_ = select(((_e8850_ * _e8942_) * ((_e8942_ * _e8942_) + (_e8954_ * _e8866_))), 0f, _e8972_);
                                                let _e8974_ = (_e8935_ <= 0f);
                                                let _e8975_ = select(((_e8851_ * _e8943_) * ((_e8943_ * _e8943_) + (_e8955_ * _e8867_))), 0f, _e8974_);
                                                let _e8977_ = (_e8860_ * _e8860_);
                                                let _e8978_ = (_e8861_ * _e8861_);
                                                let _e8979_ = (_e8862_ * _e8862_);
                                                let _e8980_ = (_e8863_ * _e8863_);
                                                let _e8981_ = (_e8864_ * _e8864_);
                                                let _e8982_ = (_e8865_ * _e8865_);
                                                let _e8983_ = (_e8866_ * _e8866_);
                                                let _e8984_ = (_e8867_ * _e8867_);
                                                let _e8985_ = (_e8977_ * _e8860_);
                                                let _e8986_ = (_e8978_ * _e8861_);
                                                let _e8987_ = (_e8979_ * _e8862_);
                                                let _e8988_ = (_e8980_ * _e8863_);
                                                let _e8989_ = (_e8981_ * _e8864_);
                                                let _e8990_ = (_e8982_ * _e8865_);
                                                let _e8991_ = (_e8983_ * _e8866_);
                                                let _e8992_ = (_e8984_ * _e8867_);
                                                let _e9117_ = select((((_e8860_ + (0.33333334f * _e8864_)) - (((((_e8977_ * _e8977_) + ((9.333333f * _e8985_) * _e8864_)) + ((14f * _e8977_) * _e8981_)) + (_e8952_ * _e8989_)) + (0.11111111f * (_e8981_ * _e8981_)))) - (3f * (((_e8977_ + ((2f * _e8860_) * _e8864_)) + (0.2f * _e8981_)) - (((_e8985_ + ((5f * _e8977_) * _e8864_)) + ((3f * _e8860_) * _e8981_)) + (0.14285715f * _e8989_))))), 0f, _e8968_);
                                                let _e9118_ = select((((_e8861_ + (0.33333334f * _e8865_)) - (((((_e8978_ * _e8978_) + ((9.333333f * _e8986_) * _e8865_)) + ((14f * _e8978_) * _e8982_)) + (_e8953_ * _e8990_)) + (0.11111111f * (_e8982_ * _e8982_)))) - (3f * (((_e8978_ + ((2f * _e8861_) * _e8865_)) + (0.2f * _e8982_)) - (((_e8986_ + ((5f * _e8978_) * _e8865_)) + ((3f * _e8861_) * _e8982_)) + (0.14285715f * _e8990_))))), 0f, _e8970_);
                                                let _e9119_ = select((((_e8862_ + (0.33333334f * _e8866_)) - (((((_e8979_ * _e8979_) + ((9.333333f * _e8987_) * _e8866_)) + ((14f * _e8979_) * _e8983_)) + (_e8954_ * _e8991_)) + (0.11111111f * (_e8983_ * _e8983_)))) - (3f * (((_e8979_ + ((2f * _e8862_) * _e8866_)) + (0.2f * _e8983_)) - (((_e8987_ + ((5f * _e8979_) * _e8866_)) + ((3f * _e8862_) * _e8983_)) + (0.14285715f * _e8991_))))), 0f, _e8972_);
                                                let _e9120_ = select((((_e8863_ + (0.33333334f * _e8867_)) - (((((_e8980_ * _e8980_) + ((9.333333f * _e8988_) * _e8867_)) + ((14f * _e8980_) * _e8984_)) + (_e8955_ * _e8992_)) + (0.11111111f * (_e8984_ * _e8984_)))) - (3f * (((_e8980_ + ((2f * _e8863_) * _e8867_)) + (0.2f * _e8984_)) - (((_e8988_ + ((5f * _e8980_) * _e8867_)) + ((3f * _e8863_) * _e8984_)) + (0.14285715f * _e8992_))))), 0f, _e8974_);
                                                if (_e8920_ <= 1f) {
                                                    if (_e8925_ <= 1f) {
                                                        let _e9124_ = (_e8930_ <= 1f);
                                                        if _e9124_ {
                                                            phi_22400_ = (_e8935_ <= 1f);
                                                        } else {
                                                            phi_22400_ = bool();
                                                        }
                                                        let _e9127_ = phi_22400_;
                                                        phi_22402_ = _e9127_;
                                                        phi_22403_ = select(true, false, _e9124_);
                                                    } else {
                                                        phi_22402_ = bool();
                                                        phi_22403_ = true;
                                                    }
                                                    let _e9130_ = phi_22402_;
                                                    let _e9132_ = phi_22403_;
                                                    phi_22404_ = _e9130_;
                                                    phi_22405_ = _e9132_;
                                                } else {
                                                    phi_22404_ = bool();
                                                    phi_22405_ = true;
                                                }
                                                let _e9134_ = phi_22404_;
                                                let _e9136_ = phi_22405_;
                                                if select(_e9134_, false, _e9136_) {
                                                    phi_22837_ = type_31_(type_30_(array<f32, 4>(_e8969_, _e8971_, _e8973_, _e8975_)), type_30_(array<f32, 4>(_e9117_, _e9118_, _e9119_, _e9120_)));
                                                } else {
                                                    if (_e8888_ != _e8888_) {
                                                        phi_33886_ = true;
                                                    } else {
                                                        phi_33886_ = (0.000001f >= _e8888_);
                                                    }
                                                    let _e9141_ = phi_33886_;
                                                    let _e9142_ = select(_e8888_, 0.000001f, _e9141_);
                                                    if (_e8889_ != _e8889_) {
                                                        phi_33901_ = true;
                                                    } else {
                                                        phi_33901_ = (0.000001f >= _e8889_);
                                                    }
                                                    let _e9146_ = phi_33901_;
                                                    let _e9147_ = select(_e8889_, 0.000001f, _e9146_);
                                                    if (_e8890_ != _e8890_) {
                                                        phi_33916_ = true;
                                                    } else {
                                                        phi_33916_ = (0.000001f >= _e8890_);
                                                    }
                                                    let _e9151_ = phi_33916_;
                                                    let _e9152_ = select(_e8890_, 0.000001f, _e9151_);
                                                    if (_e8891_ != _e8891_) {
                                                        phi_33931_ = true;
                                                    } else {
                                                        phi_33931_ = (0.000001f >= _e8891_);
                                                    }
                                                    let _e9156_ = phi_33931_;
                                                    let _e9157_ = select(_e8891_, 0.000001f, _e9156_);
                                                    let _e9160_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9142_)));
                                                    let _e9163_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9147_)));
                                                    let _e9166_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9152_)));
                                                    let _e9169_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9157_)));
                                                    let _e9179_ = (_e9160_ + (_e9160_ * (1f - (_e9142_ * _e9160_))));
                                                    let _e9181_ = (_e9163_ + (_e9163_ * (1f - (_e9147_ * _e9163_))));
                                                    let _e9183_ = (_e9166_ + (_e9166_ * (1f - (_e9152_ * _e9166_))));
                                                    let _e9185_ = (_e9169_ + (_e9169_ * (1f - (_e9157_ * _e9169_))));
                                                    let _e9195_ = (_e9179_ + (_e9179_ * (1f - (_e9142_ * _e9179_))));
                                                    let _e9197_ = (_e9181_ + (_e9181_ * (1f - (_e9147_ * _e9181_))));
                                                    let _e9199_ = (_e9183_ + (_e9183_ * (1f - (_e9152_ * _e9183_))));
                                                    let _e9201_ = (_e9185_ + (_e9185_ * (1f - (_e9157_ * _e9185_))));
                                                    let _e9238_ = (select((select(0f, 1f, (-1f <= _e8732_)) * select(0f, 1f, (_e8751_ <= 1f))), 1f, (_e8888_ <= 0.000001f)) <= 0f);
                                                    let _e9240_ = (select((select(0f, 1f, (-1f <= _e8737_)) * select(0f, 1f, (_e8755_ <= 1f))), 1f, (_e8889_ <= 0.000001f)) <= 0f);
                                                    let _e9242_ = (select((select(0f, 1f, (-1f <= _e8742_)) * select(0f, 1f, (_e8759_ <= 1f))), 1f, (_e8890_ <= 0.000001f)) <= 0f);
                                                    let _e9244_ = (select((select(0f, 1f, (-1f <= _e8747_)) * select(0f, 1f, (_e8763_ <= 1f))), 1f, (_e8891_ <= 0.000001f)) <= 0f);
                                                    let _e9246_ = (1f <= _e8732_);
                                                    let _e9248_ = (1f <= _e8737_);
                                                    let _e9250_ = (1f <= _e8742_);
                                                    let _e9252_ = (1f <= _e8747_);
                                                    let _e9254_ = (_e8751_ <= -1f);
                                                    let _e9256_ = (_e8755_ <= -1f);
                                                    let _e9258_ = (_e8759_ <= -1f);
                                                    let _e9260_ = (_e8763_ <= -1f);
                                                    phi_22837_ = type_31_(type_30_(array<f32, 4>(select(select(select(_e8969_, ((_e8872_ * _e8969_) * _e9195_), _e9238_), 0f, _e9246_), 0f, _e9254_), select(select(select(_e8971_, ((_e8877_ * _e8971_) * _e9197_), _e9240_), 0f, _e9248_), 0f, _e9256_), select(select(select(_e8973_, ((_e8882_ * _e8973_) * _e9199_), _e9242_), 0f, _e9250_), 0f, _e9258_), select(select(select(_e8975_, ((_e8887_ * _e8975_) * _e9201_), _e9244_), 0f, _e9252_), 0f, _e9260_))), type_30_(array<f32, 4>(select(select(select(_e9117_, ((_e8872_ * _e9117_) * _e9195_), _e9238_), 0f, _e9246_), 0f, _e9254_), select(select(select(_e9118_, ((_e8877_ * _e9118_) * _e9197_), _e9240_), 0f, _e9248_), 0f, _e9256_), select(select(select(_e9119_, ((_e8882_ * _e9119_) * _e9199_), _e9242_), 0f, _e9250_), 0f, _e9258_), select(select(select(_e9120_, ((_e8887_ * _e9120_) * _e9201_), _e9244_), 0f, _e9252_), 0f, _e9260_))));
                                                }
                                                let _e9291_ = phi_22837_;
                                                let _e9292_ = (1f - _e495_);
                                                let _e9293_ = (1f - _e515_);
                                                let _e9294_ = (1f - _e535_);
                                                let _e9295_ = (1f - _e555_);
                                                phi_22891_ = type_30_(array<f32, 4>(((_e8002_.member.member[0] * _e9292_) + (_e495_ * _e9291_.member.member[0])), ((_e8002_.member.member[1] * _e9293_) + (_e515_ * _e9291_.member.member[1])), ((_e8002_.member.member[2] * _e9294_) + (_e535_ * _e9291_.member.member[2])), ((_e8002_.member.member[3] * _e9295_) + (_e555_ * _e9291_.member.member[3]))));
                                                phi_22892_ = type_30_(array<f32, 4>(((_e7342_.member.member[0] * _e9292_) + (_e495_ * _e8655_.member.member[0])), ((_e7342_.member.member[1] * _e9293_) + (_e515_ * _e8655_.member.member[1])), ((_e7342_.member.member[2] * _e9294_) + (_e535_ * _e8655_.member.member[2])), ((_e7342_.member.member[3] * _e9295_) + (_e555_ * _e8655_.member.member[3]))));
                                            }
                                            let _e9359_ = phi_22891_;
                                            let _e9361_ = phi_22892_;
                                            let _e9422_ = (_e1592_ * 6u);
                                            let _e9423_ = local_9_;
                                            let _e9428_ = bitcast<u32>(_e9423_[0]);
                                            let _e9429_ = (_e9428_ & 2139095040u);
                                            if (_e9429_ == 2139095040u) {
                                                phi_23048_ = true;
                                            } else {
                                                phi_23048_ = (_e9429_ == 0u);
                                            }
                                            let _e9433_ = phi_23048_;
                                            let _e9436_ = bitcast<u32>(_e9423_[1]);
                                            let _e9437_ = (_e9436_ & 2139095040u);
                                            if (_e9437_ == 2139095040u) {
                                                phi_23064_ = true;
                                            } else {
                                                phi_23064_ = (_e9437_ == 0u);
                                            }
                                            let _e9441_ = phi_23064_;
                                            let _e9444_ = bitcast<u32>(_e9423_[2]);
                                            let _e9445_ = (_e9444_ & 2139095040u);
                                            if (_e9445_ == 2139095040u) {
                                                phi_23080_ = true;
                                            } else {
                                                phi_23080_ = (_e9445_ == 0u);
                                            }
                                            let _e9449_ = phi_23080_;
                                            let _e9452_ = bitcast<u32>(_e9423_[3]);
                                            let _e9453_ = (_e9452_ & 2139095040u);
                                            if (_e9453_ == 2139095040u) {
                                                phi_23096_ = true;
                                            } else {
                                                phi_23096_ = (_e9453_ == 0u);
                                            }
                                            let _e9457_ = phi_23096_;
                                            let _e9461_ = (_e9422_ < _e168_);
                                            if _e9461_ {
                                                if _e9461_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9422_] = vec4<f32>(bitcast<f32>(select(_e9428_, 0u, _e9433_)), bitcast<f32>(select(_e9436_, 0u, _e9441_)), bitcast<f32>(select(_e9444_, 0u, _e9449_)), bitcast<f32>(select(_e9452_, 0u, _e9457_)));
                                            }
                                            let _e9464_ = (_e9422_ + 1u);
                                            let _e9465_ = local_10_;
                                            let _e9470_ = bitcast<u32>(_e9465_[0]);
                                            let _e9471_ = (_e9470_ & 2139095040u);
                                            if (_e9471_ == 2139095040u) {
                                                phi_23134_ = true;
                                            } else {
                                                phi_23134_ = (_e9471_ == 0u);
                                            }
                                            let _e9475_ = phi_23134_;
                                            let _e9478_ = bitcast<u32>(_e9465_[1]);
                                            let _e9479_ = (_e9478_ & 2139095040u);
                                            if (_e9479_ == 2139095040u) {
                                                phi_23150_ = true;
                                            } else {
                                                phi_23150_ = (_e9479_ == 0u);
                                            }
                                            let _e9483_ = phi_23150_;
                                            let _e9486_ = bitcast<u32>(_e9465_[2]);
                                            let _e9487_ = (_e9486_ & 2139095040u);
                                            if (_e9487_ == 2139095040u) {
                                                phi_23166_ = true;
                                            } else {
                                                phi_23166_ = (_e9487_ == 0u);
                                            }
                                            let _e9491_ = phi_23166_;
                                            let _e9494_ = bitcast<u32>(_e9465_[3]);
                                            let _e9495_ = (_e9494_ & 2139095040u);
                                            if (_e9495_ == 2139095040u) {
                                                phi_23182_ = true;
                                            } else {
                                                phi_23182_ = (_e9495_ == 0u);
                                            }
                                            let _e9499_ = phi_23182_;
                                            let _e9503_ = (_e9464_ < _e168_);
                                            if _e9503_ {
                                                if _e9503_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9464_] = vec4<f32>(bitcast<f32>(select(_e9470_, 0u, _e9475_)), bitcast<f32>(select(_e9478_, 0u, _e9483_)), bitcast<f32>(select(_e9486_, 0u, _e9491_)), bitcast<f32>(select(_e9494_, 0u, _e9499_)));
                                            }
                                            let _e9506_ = (_e9422_ + 2u);
                                            let _e9511_ = bitcast<u32>(_e9361_.member[0]);
                                            let _e9512_ = (_e9511_ & 2139095040u);
                                            if (_e9512_ == 2139095040u) {
                                                phi_23214_ = true;
                                            } else {
                                                phi_23214_ = (_e9512_ == 0u);
                                            }
                                            let _e9516_ = phi_23214_;
                                            let _e9519_ = bitcast<u32>(_e9361_.member[1]);
                                            let _e9520_ = (_e9519_ & 2139095040u);
                                            if (_e9520_ == 2139095040u) {
                                                phi_23230_ = true;
                                            } else {
                                                phi_23230_ = (_e9520_ == 0u);
                                            }
                                            let _e9524_ = phi_23230_;
                                            let _e9527_ = bitcast<u32>(_e9361_.member[2]);
                                            let _e9528_ = (_e9527_ & 2139095040u);
                                            if (_e9528_ == 2139095040u) {
                                                phi_23246_ = true;
                                            } else {
                                                phi_23246_ = (_e9528_ == 0u);
                                            }
                                            let _e9532_ = phi_23246_;
                                            let _e9535_ = bitcast<u32>(_e9361_.member[3]);
                                            let _e9536_ = (_e9535_ & 2139095040u);
                                            if (_e9536_ == 2139095040u) {
                                                phi_23262_ = true;
                                            } else {
                                                phi_23262_ = (_e9536_ == 0u);
                                            }
                                            let _e9540_ = phi_23262_;
                                            let _e9544_ = (_e9506_ < _e168_);
                                            if _e9544_ {
                                                if _e9544_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9506_] = vec4<f32>(bitcast<f32>(select(_e9511_, 0u, _e9516_)), bitcast<f32>(select(_e9519_, 0u, _e9524_)), bitcast<f32>(select(_e9527_, 0u, _e9532_)), bitcast<f32>(select(_e9535_, 0u, _e9540_)));
                                            }
                                            let _e9547_ = (_e9422_ + 3u);
                                            let _e9552_ = bitcast<u32>(_e9359_.member[0]);
                                            let _e9553_ = (_e9552_ & 2139095040u);
                                            if (_e9553_ == 2139095040u) {
                                                phi_23294_ = true;
                                            } else {
                                                phi_23294_ = (_e9553_ == 0u);
                                            }
                                            let _e9557_ = phi_23294_;
                                            let _e9560_ = bitcast<u32>(_e9359_.member[1]);
                                            let _e9561_ = (_e9560_ & 2139095040u);
                                            if (_e9561_ == 2139095040u) {
                                                phi_23310_ = true;
                                            } else {
                                                phi_23310_ = (_e9561_ == 0u);
                                            }
                                            let _e9565_ = phi_23310_;
                                            let _e9568_ = bitcast<u32>(_e9359_.member[2]);
                                            let _e9569_ = (_e9568_ & 2139095040u);
                                            if (_e9569_ == 2139095040u) {
                                                phi_23326_ = true;
                                            } else {
                                                phi_23326_ = (_e9569_ == 0u);
                                            }
                                            let _e9573_ = phi_23326_;
                                            let _e9576_ = bitcast<u32>(_e9359_.member[3]);
                                            let _e9577_ = (_e9576_ & 2139095040u);
                                            if (_e9577_ == 2139095040u) {
                                                phi_23342_ = true;
                                            } else {
                                                phi_23342_ = (_e9577_ == 0u);
                                            }
                                            let _e9581_ = phi_23342_;
                                            let _e9585_ = (_e9547_ < _e168_);
                                            if _e9585_ {
                                                if _e9585_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9547_] = vec4<f32>(bitcast<f32>(select(_e9552_, 0u, _e9557_)), bitcast<f32>(select(_e9560_, 0u, _e9565_)), bitcast<f32>(select(_e9568_, 0u, _e9573_)), bitcast<f32>(select(_e9576_, 0u, _e9581_)));
                                            }
                                            let _e9588_ = (_e9422_ + 4u);
                                            let _e9589_ = bitcast<u32>((_e7342_.member_1_.member[0] + (_e466_.x * (_e9361_.member[0] * _e9361_.member[1]))));
                                            let _e9590_ = (_e9589_ & 2139095040u);
                                            if (_e9590_ == 2139095040u) {
                                                phi_23374_ = true;
                                            } else {
                                                phi_23374_ = (_e9590_ == 0u);
                                            }
                                            let _e9594_ = phi_23374_;
                                            let _e9597_ = bitcast<u32>((_e7342_.member_1_.member[1] + (_e466_.y * (_e9361_.member[1] * _e9361_.member[2]))));
                                            let _e9598_ = (_e9597_ & 2139095040u);
                                            if (_e9598_ == 2139095040u) {
                                                phi_23390_ = true;
                                            } else {
                                                phi_23390_ = (_e9598_ == 0u);
                                            }
                                            let _e9602_ = phi_23390_;
                                            let _e9605_ = bitcast<u32>((_e7342_.member_1_.member[2] + (_e466_.z * (_e9361_.member[2] * _e9361_.member[3]))));
                                            let _e9606_ = (_e9605_ & 2139095040u);
                                            if (_e9606_ == 2139095040u) {
                                                phi_23406_ = true;
                                            } else {
                                                phi_23406_ = (_e9606_ == 0u);
                                            }
                                            let _e9610_ = phi_23406_;
                                            let _e9613_ = bitcast<u32>((_e7342_.member_1_.member[3] + (_e466_.w * (_e9361_.member[3] * _e9361_.member[0]))));
                                            let _e9614_ = (_e9613_ & 2139095040u);
                                            if (_e9614_ == 2139095040u) {
                                                phi_23422_ = true;
                                            } else {
                                                phi_23422_ = (_e9614_ == 0u);
                                            }
                                            let _e9618_ = phi_23422_;
                                            let _e9622_ = (_e9588_ < _e168_);
                                            if _e9622_ {
                                                if _e9622_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9588_] = vec4<f32>(bitcast<f32>(select(_e9589_, 0u, _e9594_)), bitcast<f32>(select(_e9597_, 0u, _e9602_)), bitcast<f32>(select(_e9605_, 0u, _e9610_)), bitcast<f32>(select(_e9613_, 0u, _e9618_)));
                                            }
                                            let _e9625_ = (_e9422_ + 5u);
                                            let _e9626_ = bitcast<u32>((_e8002_.member_1_.member[0] + (_e466_.x * (_e9359_.member[0] * _e9359_.member[1]))));
                                            let _e9627_ = (_e9626_ & 2139095040u);
                                            if (_e9627_ == 2139095040u) {
                                                phi_23454_ = true;
                                            } else {
                                                phi_23454_ = (_e9627_ == 0u);
                                            }
                                            let _e9631_ = phi_23454_;
                                            let _e9634_ = bitcast<u32>((_e8002_.member_1_.member[1] + (_e466_.y * (_e9359_.member[1] * _e9359_.member[2]))));
                                            let _e9635_ = (_e9634_ & 2139095040u);
                                            if (_e9635_ == 2139095040u) {
                                                phi_23470_ = true;
                                            } else {
                                                phi_23470_ = (_e9635_ == 0u);
                                            }
                                            let _e9639_ = phi_23470_;
                                            let _e9642_ = bitcast<u32>((_e8002_.member_1_.member[2] + (_e466_.z * (_e9359_.member[2] * _e9359_.member[3]))));
                                            let _e9643_ = (_e9642_ & 2139095040u);
                                            if (_e9643_ == 2139095040u) {
                                                phi_23486_ = true;
                                            } else {
                                                phi_23486_ = (_e9643_ == 0u);
                                            }
                                            let _e9647_ = phi_23486_;
                                            let _e9650_ = bitcast<u32>((_e8002_.member_1_.member[3] + (_e466_.w * (_e9359_.member[3] * _e9359_.member[0]))));
                                            let _e9651_ = (_e9650_ & 2139095040u);
                                            if (_e9651_ == 2139095040u) {
                                                phi_23502_ = true;
                                            } else {
                                                phi_23502_ = (_e9651_ == 0u);
                                            }
                                            let _e9655_ = phi_23502_;
                                            let _e9659_ = (_e9625_ < _e168_);
                                            if _e9659_ {
                                                if _e9659_ {
                                                } else {
                                                    phi_36576_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e9625_] = vec4<f32>(bitcast<f32>(select(_e9626_, 0u, _e9631_)), bitcast<f32>(select(_e9634_, 0u, _e9639_)), bitcast<f32>(select(_e9642_, 0u, _e9647_)), bitcast<f32>(select(_e9650_, 0u, _e9655_)));
                                            }
                                            if ((_e170_ + 1u) == _e1587_) {
                                                phi_23526_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e9665_ = phi_23526_;
                                                    if (_e9665_.member < _e9665_.member_1_) {
                                                        phi_23527_ = type_15_((_e9665_.member + 1u), _e9665_.member_1_);
                                                        phi_23550_ = type_15_(1u, _e9665_.member);
                                                    } else {
                                                        phi_23527_ = _e9665_;
                                                        phi_23550_ = type_15_(0u, u32());
                                                    }
                                                    let _e9678_ = phi_23527_;
                                                    let _e9680_ = phi_23550_;
                                                    let _e9684_ = (bitcast<i32>(_e9680_.member) != 0i);
                                                    if _e9684_ {
                                                        let _e9685_ = (_e9680_.member_1_ < 4u);
                                                        if _e9685_ {
                                                        } else {
                                                            phi_35708_ = true;
                                                            break;
                                                        }
                                                        let _e9687_ = local_11_[_e9680_.member_1_];
                                                        let _e9688_ = bitcast<u32>(_e9687_);
                                                        let _e9689_ = (_e9688_ & 2139095040u);
                                                        if (_e9689_ == 2139095040u) {
                                                            phi_23575_ = true;
                                                        } else {
                                                            phi_23575_ = (_e9689_ == 0u);
                                                        }
                                                        let _e9693_ = phi_23575_;
                                                        let _e9695_ = bitcast<f32>(select(_e9688_, 0u, _e9693_));
                                                        if (_e9695_ < -1000000000000000000f) {
                                                            phi_23592_ = -1000000000000000000f;
                                                        } else {
                                                            phi_23592_ = select(_e9695_, 1000000000000000000f, (_e9695_ > 1000000000000000000f));
                                                        }
                                                        let _e9700_ = phi_23592_;
                                                        let _e9701_ = (24u + _e9680_.member_1_);
                                                        if (_e9701_ < 256u) {
                                                        } else {
                                                            phi_35708_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e9701_] = _e9700_;
                                                        if _e9685_ {
                                                        } else {
                                                            phi_35708_ = true;
                                                            break;
                                                        }
                                                        let _e9705_ = local_12_[_e9680_.member_1_];
                                                        let _e9706_ = bitcast<u32>(_e9705_);
                                                        let _e9707_ = (_e9706_ & 2139095040u);
                                                        if (_e9707_ == 2139095040u) {
                                                            phi_23617_ = true;
                                                        } else {
                                                            phi_23617_ = (_e9707_ == 0u);
                                                        }
                                                        let _e9711_ = phi_23617_;
                                                        let _e9713_ = bitcast<f32>(select(_e9706_, 0u, _e9711_));
                                                        if (_e9713_ < -1000000000000000000f) {
                                                            phi_23634_ = -1000000000000000000f;
                                                        } else {
                                                            phi_23634_ = select(_e9713_, 1000000000000000000f, (_e9713_ > 1000000000000000000f));
                                                        }
                                                        let _e9718_ = phi_23634_;
                                                        let _e9719_ = (28u + _e9680_.member_1_);
                                                        if (_e9719_ < 256u) {
                                                        } else {
                                                            phi_35708_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e9719_] = _e9718_;
                                                    }
                                                    phi_23526_ = _e9678_;
                                                    phi_35708_ = _e6663_;
                                                    if !(_e9684_) {
                                                        break;
                                                    }
                                                }
                                                let _e9724_ = phi_35708_;
                                                phi_36576_ = _e9724_;
                                                if _e9724_ {
                                                    break;
                                                }
                                                phi_36581_ = _e9724_;
                                            } else {
                                                phi_36581_ = _e6663_;
                                            }
                                            let _e9726_ = phi_36581_;
                                            phi_36580_ = _e9726_;
                                        } else {
                                            phi_36580_ = _e6586_;
                                        }
                                        let _e9728_ = phi_36580_;
                                        phi_36579_ = _e9728_;
                                    } else {
                                        phi_36579_ = _e6586_;
                                    }
                                    let _e9730_ = phi_36579_;
                                    workgroupBarrier();
                                    phi_36578_ = _e9730_;
                                } else {
                                    phi_36578_ = _e1558_;
                                }
                                let _e9732_ = phi_36578_;
                                phi_36577_ = _e9732_;
                            } else {
                                phi_36577_ = _e1558_;
                            }
                            let _e9734_ = phi_36577_;
                            phi_34159_ = _e9734_;
                            phi_2851_ = _e1573_;
                            phi_36576_ = _e9734_;
                            if !(_e1579_) {
                                break;
                            }
                        }
                        let _e9737_ = phi_36576_;
                        phi_36582_ = _e9737_;
                        if _e9737_ {
                            break;
                        }
                        phi_36583_ = _e9737_;
                    } else {
                        phi_36583_ = _e1167_;
                    }
                    let _e9739_ = phi_36583_;
                    phi_34162_ = _e9739_;
                    phi_2126_ = _e1182_;
                    phi_36582_ = _e9739_;
                    if !(_e1188_) {
                        break;
                    }
                }
                let _e9742_ = phi_36582_;
                if _e9742_ {
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
