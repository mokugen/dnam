struct type_5_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: f32,
    member_5_: f32,
    member_6_: f32,
    member_7_: f32,
}

struct type_6_ {
    member: type_5_,
}

struct type_10_ {
    member: array<vec4<f32>>,
}

struct type_20_ {
    member: u32,
    member_1_: u32,
}

struct type_27_ {
    member: f32,
    member_1_: f32,
}

struct type_30_ {
    member: array<vec2<f32>>,
}

struct type_32_ {
    member: i32,
    member_1_: i32,
}

struct type_33_ {
    member: u32,
    member_1_: i32,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
@group(0) @binding(0) 
var<uniform> uniforms: type_6_;
@group(0) @binding(1) 
var<storage> pred_audio: type_30_;
@group(0) @binding(2) 
var<storage> target_audio: type_30_;
@group(0) @binding(3) 
var<storage> bezout_taps: type_10_;
@group(0) @binding(4) 
var<storage, read_write> d_loss_d_pred: type_30_;
@group(0) @binding(5) 
var<storage, read_write> band_losses: type_10_;
var<workgroup> buf_a: array<vec4<f32>, 256>;
var<workgroup> buf_b: array<vec4<f32>, 256>;
var<workgroup> buf_c: array<vec2<f32>, 256>;
var<workgroup> reduce_buf: array<f32, 256>;

fn function_() {
    var phi_2126_: u32;
    var phi_2129_: u32;
    var phi_2143_: u32;
    var phi_2144_: u32;
    var phi_2127_: u32;
    var phi_2130_: u32;
    var phi_2145_: bool;
    var phi_2173_: u32;
    var phi_2176_: u32;
    var phi_2174_: u32;
    var phi_2177_: u32;
    var phi_2188_: u32;
    var phi_2191_: u32;
    var phi_2193_: u32;
    var phi_2195_: u32;
    var phi_2209_: u32;
    var phi_2210_: u32;
    var phi_2189_: u32;
    var phi_2192_: u32;
    var phi_2194_: u32;
    var phi_2196_: u32;
    var phi_2213_: u32;
    var phi_2214_: u32;
    var local: u32;
    var local_1_: u32;
    var local_2_: u32;
    var phi_2239_: u32;
    var local_3_: u32;
    var phi_2274_: bool;
    var phi_2280_: f32;
    var phi_2325_: u32;
    var phi_2348_: u32;
    var phi_2373_: bool;
    var phi_2391_: bool;
    var phi_2408_: bool;
    var phi_2425_: bool;
    var phi_2431_: f32;
    var phi_2432_: f32;
    var phi_2433_: f32;
    var phi_2434_: f32;
    var phi_2436_: f32;
    var phi_2437_: f32;
    var phi_2438_: f32;
    var phi_2439_: f32;
    var phi_2440_: u32;
    var phi_2441_: bool;
    var phi_2445_: bool;
    var phi_2446_: f32;
    var phi_2447_: f32;
    var phi_2448_: f32;
    var phi_2449_: f32;
    var phi_2450_: u32;
    var phi_2451_: bool;
    var phi_2454_: bool;
    var phi_2455_: bool;
    var phi_2461_: u32;
    var phi_2478_: u32;
    var phi_2481_: u32;
    var phi_2483_: u32;
    var local_4_: u32;
    var phi_2479_: u32;
    var phi_2482_: u32;
    var phi_2484_: u32;
    var phi_2496_: u32;
    var phi_5379_: bool;
    var phi_2534_: u32;
    var local_5_: u32;
    var phi_2643_: f32;
    var phi_2644_: f32;
    var phi_2654_: bool;
    var phi_2657_: type_32_;
    var phi_2660_: f32;
    var phi_2662_: f32;
    var phi_2664_: u32;
    var phi_2666_: u32;
    var phi_2693_: type_33_;
    var phi_2658_: type_32_;
    var phi_2707_: type_33_;
    var phi_2729_: bool;
    var phi_2731_: bool;
    var phi_2746_: f32;
    var phi_2747_: f32;
    var phi_2748_: u32;
    var phi_2749_: u32;
    var phi_2750_: f32;
    var phi_2751_: f32;
    var phi_2752_: u32;
    var phi_2753_: u32;
    var phi_2661_: f32;
    var phi_2663_: f32;
    var phi_2665_: u32;
    var phi_2667_: u32;
    var phi_2755_: bool;
    var phi_5373_: bool;
    var local_6_: f32;
    var local_7_: f32;
    var phi_5444_: bool;
    var phi_2779_: type_27_;
    var phi_5443_: bool;
    var phi_2780_: type_27_;
    var phi_2781_: bool;
    var phi_2786_: type_27_;
    var phi_5441_: bool;
    var phi_2787_: type_27_;
    var phi_5440_: bool;
    var phi_2788_: type_27_;
    var phi_2789_: bool;
    var phi_5439_: bool;
    var phi_2790_: type_27_;
    var phi_2791_: bool;
    var phi_2796_: type_27_;
    var phi_5433_: bool;
    var phi_5432_: bool;
    var phi_2535_: u32;
    var phi_5431_: bool;
    var phi_2986_: bool;
    var phi_2992_: f32;
    var phi_3007_: bool;
    var phi_3013_: f32;
    var phi_3028_: bool;
    var phi_3034_: f32;
    var phi_3064_: bool;
    var phi_3070_: f32;
    var phi_3092_: bool;
    var phi_3098_: f32;
    var phi_3110_: bool;
    var phi_3134_: u32;
    var phi_3135_: u32;
    var phi_5447_: bool;
    var phi_3182_: bool;
    var phi_3199_: bool;
    var phi_3223_: u32;
    var phi_3224_: u32;
    var phi_5474_: bool;
    var phi_3271_: bool;
    var phi_3288_: bool;
    var phi_3312_: u32;
    var phi_3313_: u32;
    var phi_5484_: bool;
    var phi_3360_: bool;
    var phi_3377_: bool;
    var phi_3401_: u32;
    var phi_3402_: u32;
    var phi_5494_: bool;
    var phi_3449_: bool;
    var phi_3466_: bool;
    var phi_3490_: u32;
    var phi_3491_: u32;
    var phi_5504_: bool;
    var phi_3538_: bool;
    var phi_3555_: bool;
    var phi_3579_: u32;
    var phi_3580_: u32;
    var phi_5514_: bool;
    var phi_3627_: bool;
    var phi_3646_: bool;
    var phi_3663_: bool;
    var phi_3681_: bool;
    var phi_3699_: bool;
    var phi_4663_: bool;
    var phi_3791_: bool;
    var phi_4723_: bool;
    var phi_3756_: bool;
    var phi_3797_: f32;
    var phi_3798_: f32;
    var phi_3799_: f32;
    var phi_3800_: bool;
    var phi_4783_: bool;
    var phi_3891_: bool;
    var phi_4843_: bool;
    var phi_3856_: bool;
    var phi_3897_: f32;
    var phi_3898_: f32;
    var phi_3899_: f32;
    var phi_3900_: bool;
    var phi_4903_: bool;
    var phi_3991_: bool;
    var phi_4963_: bool;
    var phi_3956_: bool;
    var phi_3997_: f32;
    var phi_3998_: f32;
    var phi_3999_: f32;
    var phi_4000_: bool;
    var phi_4016_: bool;
    var phi_4033_: bool;
    var phi_4050_: bool;
    var phi_4071_: bool;
    var phi_4102_: u32;
    var phi_4123_: vec4<f32>;
    var phi_5530_: bool;
    var phi_4142_: u32;
    var local_8_: u32;
    var local_9_: u32;
    var phi_4253_: f32;
    var phi_4254_: f32;
    var phi_4264_: bool;
    var phi_4267_: type_32_;
    var phi_4270_: f32;
    var phi_4272_: f32;
    var phi_4274_: u32;
    var phi_4276_: u32;
    var phi_4303_: type_33_;
    var phi_4268_: type_32_;
    var phi_4317_: type_33_;
    var phi_4339_: bool;
    var phi_4341_: bool;
    var phi_4356_: f32;
    var phi_4357_: f32;
    var phi_4358_: u32;
    var phi_4359_: u32;
    var phi_4360_: f32;
    var phi_4361_: f32;
    var phi_4362_: u32;
    var phi_4363_: u32;
    var phi_4271_: f32;
    var phi_4273_: f32;
    var phi_4275_: u32;
    var phi_4277_: u32;
    var phi_4365_: bool;
    var phi_5524_: bool;
    var local_10_: f32;
    var local_11_: f32;
    var phi_5616_: bool;
    var phi_4389_: type_27_;
    var phi_5615_: bool;
    var phi_4390_: type_27_;
    var phi_4391_: bool;
    var phi_4396_: type_27_;
    var phi_5613_: bool;
    var phi_4397_: type_27_;
    var phi_5612_: bool;
    var phi_4398_: type_27_;
    var phi_4399_: bool;
    var phi_5611_: bool;
    var phi_4400_: type_27_;
    var phi_4401_: bool;
    var phi_4406_: type_27_;
    var phi_5605_: bool;
    var phi_5604_: bool;
    var phi_4143_: u32;
    var phi_5603_: bool;
    var phi_4488_: u32;
    var phi_4491_: u32;
    var phi_4493_: u32;
    var local_12_: u32;
    var phi_4489_: u32;
    var phi_4492_: u32;
    var phi_4494_: u32;
    var local_13_: u32;
    var local_14_: u32;
    var phi_4567_: bool;
    var phi_4584_: bool;
    var phi_4600_: u32;
    var phi_174_: u32;
    var phi_181_: u32;
    var phi_204_: bool;
    var phi_221_: u32;
    var phi_244_: u32;
    var phi_268_: bool;
    var phi_285_: bool;
    var phi_301_: bool;
    var phi_317_: bool;
    var phi_363_: type_20_;
    var phi_366_: f32;
    var phi_368_: f32;
    var phi_370_: f32;
    var phi_372_: f32;
    var phi_374_: f32;
    var phi_376_: f32;
    var phi_378_: f32;
    var phi_380_: f32;
    var phi_382_: f32;
    var phi_384_: f32;
    var phi_386_: f32;
    var phi_388_: f32;
    var phi_364_: type_20_;
    var phi_411_: type_20_;
    var phi_433_: u32;
    var phi_464_: bool;
    var phi_480_: bool;
    var phi_496_: bool;
    var phi_558_: f32;
    var phi_559_: f32;
    var phi_560_: f32;
    var phi_561_: f32;
    var phi_562_: f32;
    var phi_563_: f32;
    var phi_564_: f32;
    var phi_565_: f32;
    var phi_566_: f32;
    var phi_567_: f32;
    var phi_568_: f32;
    var phi_569_: f32;
    var phi_367_: f32;
    var phi_369_: f32;
    var phi_371_: f32;
    var phi_373_: f32;
    var phi_375_: f32;
    var phi_377_: f32;
    var phi_379_: f32;
    var phi_381_: f32;
    var phi_383_: f32;
    var phi_385_: f32;
    var phi_387_: f32;
    var phi_389_: f32;
    var phi_5619_: bool;
    var phi_5675_: bool;
    var phi_570_: f32;
    var phi_571_: f32;
    var phi_572_: f32;
    var phi_573_: f32;
    var phi_574_: f32;
    var phi_575_: f32;
    var phi_576_: f32;
    var phi_577_: f32;
    var phi_578_: f32;
    var phi_579_: f32;
    var phi_580_: f32;
    var phi_581_: f32;
    var phi_635_: bool;
    var phi_641_: f32;
    var phi_658_: bool;
    var phi_664_: f32;
    var phi_681_: bool;
    var phi_687_: f32;
    var phi_704_: bool;
    var phi_710_: f32;
    var phi_727_: bool;
    var phi_733_: f32;
    var phi_750_: bool;
    var phi_756_: f32;
    var phi_767_: bool;
    var phi_791_: u32;
    var phi_792_: u32;
    var phi_5658_: bool;
    var phi_839_: bool;
    var phi_855_: bool;
    var phi_879_: u32;
    var phi_880_: u32;
    var phi_5689_: bool;
    var phi_927_: bool;
    var phi_943_: bool;
    var phi_967_: u32;
    var phi_968_: u32;
    var phi_5699_: bool;
    var phi_1015_: bool;
    var phi_1031_: bool;
    var phi_1055_: u32;
    var phi_1056_: u32;
    var phi_5709_: bool;
    var phi_1103_: bool;
    var phi_1119_: bool;
    var phi_1143_: u32;
    var phi_1144_: u32;
    var phi_5719_: bool;
    var phi_1191_: bool;
    var phi_1207_: bool;
    var phi_1231_: u32;
    var phi_1232_: u32;
    var phi_5729_: bool;
    var phi_1279_: bool;
    var phi_1298_: bool;
    var phi_1315_: bool;
    var phi_1333_: bool;
    var phi_1351_: bool;
    var phi_5023_: bool;
    var phi_1443_: bool;
    var phi_5083_: bool;
    var phi_1408_: bool;
    var phi_1449_: f32;
    var phi_1450_: f32;
    var phi_1451_: f32;
    var phi_1452_: bool;
    var phi_5143_: bool;
    var phi_1543_: bool;
    var phi_5203_: bool;
    var phi_1508_: bool;
    var phi_1549_: f32;
    var phi_1550_: f32;
    var phi_1551_: f32;
    var phi_1552_: bool;
    var phi_5263_: bool;
    var phi_1643_: bool;
    var phi_5323_: bool;
    var phi_1608_: bool;
    var phi_1649_: f32;
    var phi_1650_: f32;
    var phi_1651_: f32;
    var phi_1652_: bool;
    var phi_1668_: bool;
    var phi_1685_: bool;
    var phi_1702_: bool;
    var phi_1723_: bool;
    var phi_1754_: u32;
    var phi_1836_: type_20_;
    var phi_1839_: f32;
    var phi_1841_: f32;
    var phi_1837_: type_20_;
    var phi_1864_: type_20_;
    var phi_1887_: u32;
    var phi_1972_: bool;
    var phi_1989_: bool;
    var phi_2007_: bool;
    var phi_2026_: bool;
    var phi_2043_: bool;
    var phi_2061_: bool;
    var phi_2070_: f32;
    var phi_2071_: f32;
    var phi_1840_: f32;
    var phi_1842_: f32;
    var phi_5739_: bool;
    var local_15_: f32;
    var phi_2082_: bool;
    var local_16_: f32;
    var phi_2098_: bool;
    var phi_2114_: u32;
    var local_17_: u32;
    var local_18_: u32;
    var local_19_: u32;
    var local_20_: u32;
    var local_21_: f32;
    var local_22_: f32;
    var local_23_: f32;
    var local_24_: f32;
    var local_25_: f32;
    var local_26_: f32;
    var local_27_: f32;
    var local_28_: f32;
    var local_29_: f32;
    var local_30_: f32;
    var local_31_: f32;
    var local_32_: f32;

    switch bitcast<i32>(0u) {
        default: {
            let _e80_ = wg_id_1_;
            let _e81_ = lid_1_;
            let _e83_ = arrayLength((&pred_audio.member));
            let _e85_ = arrayLength((&target_audio.member));
            let _e87_ = arrayLength((&bezout_taps.member));
            let _e89_ = arrayLength((&d_loss_d_pred.member));
            let _e91_ = arrayLength((&band_losses.member));
            let _e96_ = uniforms.member.member_1_;
            if (_e96_ == 0u) {
                let _e1686_ = uniforms.member.member;
                let _e1689_ = uniforms.member.member_3_;
                let _e1691_ = select(1u, _e1689_, (1u < _e1689_));
                if (_e1686_ == 0u) {
                    phi_174_ = 1u;
                } else {
                    phi_174_ = (((_e1686_ - 1u) >> bitcast<u32>(8i)) + 1u);
                }
                let _e1698_ = phi_174_;
                let _e1699_ = (_e80_.x * 256u);
                if (_e1686_ > _e1699_) {
                    phi_181_ = (_e1686_ - _e1699_);
                } else {
                    phi_181_ = 0u;
                }
                let _e1703_ = phi_181_;
                let _e1705_ = select(_e1703_, 256u, (256u < _e1703_));
                let _e1708_ = uniforms.member.member_2_;
                let _e1710_ = select(1u, _e1708_, (1u < _e1708_));
                let _e1712_ = select(_e1710_, 64u, (64u < _e1710_));
                if (_e81_ < _e1705_) {
                    phi_204_ = (_e80_.y < _e1691_);
                } else {
                    phi_204_ = false;
                }
                let _e1716_ = phi_204_;
                let _e1719_ = ((_e80_.y * _e1686_) + (_e1699_ + _e81_));
                if _e1716_ {
                    if (_e83_ == 0u) {
                        phi_221_ = 0u;
                    } else {
                        let _e1725_ = (_e83_ - 1u);
                        phi_221_ = select(_e1725_, _e1719_, (_e1719_ < _e1725_));
                    }
                    let _e1729_ = phi_221_;
                    if (_e1729_ < _e83_) {
                    } else {
                        break;
                    }
                    let _e1734_ = pred_audio.member[_e1729_].x;
                    let _e1738_ = pred_audio.member[_e1729_].y;
                    if (_e85_ == 0u) {
                        phi_244_ = 0u;
                    } else {
                        let _e1740_ = (_e85_ - 1u);
                        phi_244_ = select(_e1740_, _e1719_, (_e1719_ < _e1740_));
                    }
                    let _e1744_ = phi_244_;
                    if (_e1744_ < _e85_) {
                    } else {
                        break;
                    }
                    let _e1749_ = target_audio.member[_e1744_].x;
                    let _e1753_ = target_audio.member[_e1744_].y;
                    let _e1755_ = bitcast<u32>((_e1734_ - _e1749_));
                    let _e1756_ = (_e1755_ & 2139095040u);
                    if (_e1756_ == 2139095040u) {
                        phi_268_ = true;
                    } else {
                        phi_268_ = (_e1756_ == 0u);
                    }
                    let _e1760_ = phi_268_;
                    let _e1764_ = bitcast<u32>((_e1738_ - _e1753_));
                    let _e1765_ = (_e1764_ & 2139095040u);
                    if (_e1765_ == 2139095040u) {
                        phi_285_ = true;
                    } else {
                        phi_285_ = (_e1765_ == 0u);
                    }
                    let _e1769_ = phi_285_;
                    let _e1772_ = bitcast<u32>(_e1749_);
                    let _e1773_ = (_e1772_ & 2139095040u);
                    if (_e1773_ == 2139095040u) {
                        phi_301_ = true;
                    } else {
                        phi_301_ = (_e1773_ == 0u);
                    }
                    let _e1777_ = phi_301_;
                    let _e1780_ = bitcast<u32>(_e1753_);
                    let _e1781_ = (_e1780_ & 2139095040u);
                    if (_e1781_ == 2139095040u) {
                        phi_317_ = true;
                    } else {
                        phi_317_ = (_e1781_ == 0u);
                    }
                    let _e1785_ = phi_317_;
                    let _e1790_ = select(255u, _e81_, (_e81_ < 255u));
                    if (_e1790_ < 256u) {
                    } else {
                        break;
                    }
                    buf_a[_e1790_] = vec4<f32>(bitcast<f32>(select(_e1755_, 0u, _e1760_)), bitcast<f32>(select(_e1764_, 0u, _e1769_)), bitcast<f32>(select(_e1772_, 0u, _e1777_)), bitcast<f32>(select(_e1780_, 0u, _e1785_)));
                } else {
                    let _e1721_ = select(255u, _e81_, (_e81_ < 255u));
                    if (_e1721_ < 256u) {
                    } else {
                        break;
                    }
                    buf_a[_e1721_] = vec4<f32>(0f, 0f, 0f, 0f);
                }
                workgroupBarrier();
                if _e1716_ {
                    phi_363_ = type_20_(0u, _e1712_);
                    phi_366_ = 0f;
                    phi_368_ = 0f;
                    phi_370_ = 0f;
                    phi_372_ = 0f;
                    phi_374_ = 0f;
                    phi_376_ = 0f;
                    phi_378_ = 0f;
                    phi_380_ = 0f;
                    phi_382_ = 0f;
                    phi_384_ = 0f;
                    phi_386_ = 0f;
                    phi_388_ = 0f;
                    loop {
                        let _e1795_ = phi_363_;
                        let _e1797_ = phi_366_;
                        let _e1799_ = phi_368_;
                        let _e1801_ = phi_370_;
                        let _e1803_ = phi_372_;
                        let _e1805_ = phi_374_;
                        let _e1807_ = phi_376_;
                        let _e1809_ = phi_378_;
                        let _e1811_ = phi_380_;
                        let _e1813_ = phi_382_;
                        let _e1815_ = phi_384_;
                        let _e1817_ = phi_386_;
                        let _e1819_ = phi_388_;
                        local_21_ = _e1797_;
                        local_22_ = _e1799_;
                        local_23_ = _e1801_;
                        local_24_ = _e1803_;
                        local_25_ = _e1805_;
                        local_26_ = _e1807_;
                        local_27_ = _e1809_;
                        local_28_ = _e1811_;
                        local_29_ = _e1813_;
                        local_30_ = _e1815_;
                        local_31_ = _e1817_;
                        local_32_ = _e1819_;
                        if (_e1795_.member < _e1795_.member_1_) {
                            phi_364_ = type_20_((_e1795_.member + 1u), _e1795_.member_1_);
                            phi_411_ = type_20_(1u, _e1795_.member);
                        } else {
                            phi_364_ = _e1795_;
                            phi_411_ = type_20_(0u, u32());
                        }
                        let _e1832_ = phi_364_;
                        let _e1834_ = phi_411_;
                        let _e1838_ = (bitcast<i32>(_e1834_.member) != 0i);
                        if _e1838_ {
                            if (_e1834_.member_1_ <= _e81_) {
                                if (_e87_ == 0u) {
                                    phi_433_ = 0u;
                                } else {
                                    let _e1841_ = (_e87_ - 1u);
                                    phi_433_ = select(_e1841_, _e1834_.member_1_, (_e1834_.member_1_ < _e1841_));
                                }
                                let _e1845_ = phi_433_;
                                if (_e1845_ < _e87_) {
                                } else {
                                    phi_5619_ = true;
                                    break;
                                }
                                let _e1850_ = bezout_taps.member[_e1845_].x;
                                let _e1854_ = bezout_taps.member[_e1845_].y;
                                let _e1858_ = bezout_taps.member[_e1845_].z;
                                let _e1859_ = bitcast<u32>(_e1850_);
                                let _e1860_ = (_e1859_ & 2139095040u);
                                if (_e1860_ == 2139095040u) {
                                    phi_464_ = true;
                                } else {
                                    phi_464_ = (_e1860_ == 0u);
                                }
                                let _e1864_ = phi_464_;
                                let _e1866_ = bitcast<f32>(select(_e1859_, 0u, _e1864_));
                                let _e1867_ = bitcast<u32>(_e1854_);
                                let _e1868_ = (_e1867_ & 2139095040u);
                                if (_e1868_ == 2139095040u) {
                                    phi_480_ = true;
                                } else {
                                    phi_480_ = (_e1868_ == 0u);
                                }
                                let _e1872_ = phi_480_;
                                let _e1874_ = bitcast<f32>(select(_e1867_, 0u, _e1872_));
                                let _e1875_ = bitcast<u32>(_e1858_);
                                let _e1876_ = (_e1875_ & 2139095040u);
                                if (_e1876_ == 2139095040u) {
                                    phi_496_ = true;
                                } else {
                                    phi_496_ = (_e1876_ == 0u);
                                }
                                let _e1880_ = phi_496_;
                                let _e1882_ = bitcast<f32>(select(_e1875_, 0u, _e1880_));
                                let _e1883_ = (_e81_ - _e1834_.member_1_);
                                let _e1885_ = select(255u, _e1883_, (_e1883_ < 255u));
                                if (_e1885_ < 256u) {
                                } else {
                                    phi_5619_ = true;
                                    break;
                                }
                                let _e1889_ = buf_a[_e1885_].x;
                                let _e1892_ = buf_a[_e1885_].y;
                                let _e1895_ = buf_a[_e1885_].z;
                                let _e1898_ = buf_a[_e1885_].w;
                                phi_558_ = (_e1797_ + (_e1882_ * _e1898_));
                                phi_559_ = (_e1799_ + (_e1882_ * _e1895_));
                                phi_560_ = (_e1801_ + (_e1874_ * _e1898_));
                                phi_561_ = (_e1803_ + (_e1874_ * _e1895_));
                                phi_562_ = (_e1805_ + (_e1866_ * _e1898_));
                                phi_563_ = (_e1807_ + (_e1866_ * _e1895_));
                                phi_564_ = (_e1809_ + (_e1882_ * _e1892_));
                                phi_565_ = (_e1811_ + (_e1882_ * _e1889_));
                                phi_566_ = (_e1813_ + (_e1874_ * _e1892_));
                                phi_567_ = (_e1815_ + (_e1874_ * _e1889_));
                                phi_568_ = (_e1817_ + (_e1866_ * _e1892_));
                                phi_569_ = (_e1819_ + (_e1866_ * _e1889_));
                            } else {
                                phi_558_ = _e1797_;
                                phi_559_ = _e1799_;
                                phi_560_ = _e1801_;
                                phi_561_ = _e1803_;
                                phi_562_ = _e1805_;
                                phi_563_ = _e1807_;
                                phi_564_ = _e1809_;
                                phi_565_ = _e1811_;
                                phi_566_ = _e1813_;
                                phi_567_ = _e1815_;
                                phi_568_ = _e1817_;
                                phi_569_ = _e1819_;
                            }
                            let _e1924_ = phi_558_;
                            let _e1926_ = phi_559_;
                            let _e1928_ = phi_560_;
                            let _e1930_ = phi_561_;
                            let _e1932_ = phi_562_;
                            let _e1934_ = phi_563_;
                            let _e1936_ = phi_564_;
                            let _e1938_ = phi_565_;
                            let _e1940_ = phi_566_;
                            let _e1942_ = phi_567_;
                            let _e1944_ = phi_568_;
                            let _e1946_ = phi_569_;
                            phi_367_ = _e1924_;
                            phi_369_ = _e1926_;
                            phi_371_ = _e1928_;
                            phi_373_ = _e1930_;
                            phi_375_ = _e1932_;
                            phi_377_ = _e1934_;
                            phi_379_ = _e1936_;
                            phi_381_ = _e1938_;
                            phi_383_ = _e1940_;
                            phi_385_ = _e1942_;
                            phi_387_ = _e1944_;
                            phi_389_ = _e1946_;
                        } else {
                            phi_367_ = f32();
                            phi_369_ = f32();
                            phi_371_ = f32();
                            phi_373_ = f32();
                            phi_375_ = f32();
                            phi_377_ = f32();
                            phi_379_ = f32();
                            phi_381_ = f32();
                            phi_383_ = f32();
                            phi_385_ = f32();
                            phi_387_ = f32();
                            phi_389_ = f32();
                        }
                        let _e1948_ = phi_367_;
                        let _e1950_ = phi_369_;
                        let _e1952_ = phi_371_;
                        let _e1954_ = phi_373_;
                        let _e1956_ = phi_375_;
                        let _e1958_ = phi_377_;
                        let _e1960_ = phi_379_;
                        let _e1962_ = phi_381_;
                        let _e1964_ = phi_383_;
                        let _e1966_ = phi_385_;
                        let _e1968_ = phi_387_;
                        let _e1970_ = phi_389_;
                        phi_363_ = _e1832_;
                        phi_366_ = _e1948_;
                        phi_368_ = _e1950_;
                        phi_370_ = _e1952_;
                        phi_372_ = _e1954_;
                        phi_374_ = _e1956_;
                        phi_376_ = _e1958_;
                        phi_378_ = _e1960_;
                        phi_380_ = _e1962_;
                        phi_382_ = _e1964_;
                        phi_384_ = _e1966_;
                        phi_386_ = _e1968_;
                        phi_388_ = _e1970_;
                        phi_5619_ = false;
                        if !(_e1838_) {
                            break;
                        }
                    }
                    let _e1973_ = phi_5619_;
                    if _e1973_ {
                        break;
                    }
                    phi_5675_ = _e1973_;
                    let _e3136_ = local_21_;
                    phi_570_ = _e3136_;
                    let _e3139_ = local_22_;
                    phi_571_ = _e3139_;
                    let _e3142_ = local_23_;
                    phi_572_ = _e3142_;
                    let _e3145_ = local_24_;
                    phi_573_ = _e3145_;
                    let _e3148_ = local_25_;
                    phi_574_ = _e3148_;
                    let _e3151_ = local_26_;
                    phi_575_ = _e3151_;
                    let _e3154_ = local_27_;
                    phi_576_ = _e3154_;
                    let _e3157_ = local_28_;
                    phi_577_ = _e3157_;
                    let _e3160_ = local_29_;
                    phi_578_ = _e3160_;
                    let _e3163_ = local_30_;
                    phi_579_ = _e3163_;
                    let _e3166_ = local_31_;
                    phi_580_ = _e3166_;
                    let _e3169_ = local_32_;
                    phi_581_ = _e3169_;
                } else {
                    phi_5675_ = false;
                    phi_570_ = 0f;
                    phi_571_ = 0f;
                    phi_572_ = 0f;
                    phi_573_ = 0f;
                    phi_574_ = 0f;
                    phi_575_ = 0f;
                    phi_576_ = 0f;
                    phi_577_ = 0f;
                    phi_578_ = 0f;
                    phi_579_ = 0f;
                    phi_580_ = 0f;
                    phi_581_ = 0f;
                }
                let _e1975_ = phi_5675_;
                let _e1977_ = phi_570_;
                let _e1979_ = phi_571_;
                let _e1981_ = phi_572_;
                let _e1983_ = phi_573_;
                let _e1985_ = phi_574_;
                let _e1987_ = phi_575_;
                let _e1989_ = phi_576_;
                let _e1991_ = phi_577_;
                let _e1993_ = phi_578_;
                let _e1995_ = phi_579_;
                let _e1997_ = phi_580_;
                let _e1999_ = phi_581_;
                let _e2002_ = select(255u, _e81_, (_e81_ < 255u));
                let _e2003_ = (_e2002_ < 256u);
                if _e2003_ {
                } else {
                    break;
                }
                buf_b[_e2002_] = vec4<f32>(_e1999_, _e1997_, _e1995_, _e1993_);
                if _e2003_ {
                } else {
                    break;
                }
                buf_c[_e2002_].x = _e1991_;
                buf_c[_e2002_].y = _e1989_;
                if _e1716_ {
                    let _e2012_ = bitcast<u32>(((_e1999_ * _e1999_) + (_e1997_ * _e1997_)));
                    let _e2013_ = (_e2012_ & 2139095040u);
                    if (_e2013_ == 2139095040u) {
                        phi_635_ = true;
                    } else {
                        phi_635_ = (_e2013_ == 0u);
                    }
                    let _e2017_ = phi_635_;
                    phi_641_ = bitcast<f32>(select(_e2012_, 0u, _e2017_));
                } else {
                    phi_641_ = 0f;
                }
                let _e2021_ = phi_641_;
                if _e1716_ {
                    let _e2025_ = bitcast<u32>(((_e1987_ * _e1987_) + (_e1985_ * _e1985_)));
                    let _e2026_ = (_e2025_ & 2139095040u);
                    if (_e2026_ == 2139095040u) {
                        phi_658_ = true;
                    } else {
                        phi_658_ = (_e2026_ == 0u);
                    }
                    let _e2030_ = phi_658_;
                    phi_664_ = bitcast<f32>(select(_e2025_, 0u, _e2030_));
                } else {
                    phi_664_ = 0f;
                }
                let _e2034_ = phi_664_;
                if _e1716_ {
                    let _e2038_ = bitcast<u32>(((_e1995_ * _e1995_) + (_e1993_ * _e1993_)));
                    let _e2039_ = (_e2038_ & 2139095040u);
                    if (_e2039_ == 2139095040u) {
                        phi_681_ = true;
                    } else {
                        phi_681_ = (_e2039_ == 0u);
                    }
                    let _e2043_ = phi_681_;
                    phi_687_ = bitcast<f32>(select(_e2038_, 0u, _e2043_));
                } else {
                    phi_687_ = 0f;
                }
                let _e2047_ = phi_687_;
                if _e1716_ {
                    let _e2051_ = bitcast<u32>(((_e1983_ * _e1983_) + (_e1981_ * _e1981_)));
                    let _e2052_ = (_e2051_ & 2139095040u);
                    if (_e2052_ == 2139095040u) {
                        phi_704_ = true;
                    } else {
                        phi_704_ = (_e2052_ == 0u);
                    }
                    let _e2056_ = phi_704_;
                    phi_710_ = bitcast<f32>(select(_e2051_, 0u, _e2056_));
                } else {
                    phi_710_ = 0f;
                }
                let _e2060_ = phi_710_;
                if _e1716_ {
                    let _e2064_ = bitcast<u32>(((_e1991_ * _e1991_) + (_e1989_ * _e1989_)));
                    let _e2065_ = (_e2064_ & 2139095040u);
                    if (_e2065_ == 2139095040u) {
                        phi_727_ = true;
                    } else {
                        phi_727_ = (_e2065_ == 0u);
                    }
                    let _e2069_ = phi_727_;
                    phi_733_ = bitcast<f32>(select(_e2064_, 0u, _e2069_));
                } else {
                    phi_733_ = 0f;
                }
                let _e2073_ = phi_733_;
                if _e1716_ {
                    let _e2077_ = bitcast<u32>(((_e1979_ * _e1979_) + (_e1977_ * _e1977_)));
                    let _e2078_ = (_e2077_ & 2139095040u);
                    if (_e2078_ == 2139095040u) {
                        phi_750_ = true;
                    } else {
                        phi_750_ = (_e2078_ == 0u);
                    }
                    let _e2082_ = phi_750_;
                    phi_756_ = bitcast<f32>(select(_e2077_, 0u, _e2082_));
                } else {
                    phi_756_ = 0f;
                }
                let _e2086_ = phi_756_;
                let _e2087_ = bitcast<u32>(_e2021_);
                let _e2088_ = (_e2087_ & 2139095040u);
                if (_e2088_ == 2139095040u) {
                    phi_767_ = true;
                } else {
                    phi_767_ = (_e2088_ == 0u);
                }
                let _e2092_ = phi_767_;
                if _e2003_ {
                } else {
                    break;
                }
                reduce_buf[_e2002_] = bitcast<f32>(select(_e2087_, 0u, _e2092_));
                workgroupBarrier();
                phi_791_ = 128u;
                loop {
                    let _e2097_ = phi_791_;
                    let _e2098_ = (_e2097_ > 0u);
                    if _e2098_ {
                        if (_e81_ < _e2097_) {
                            let _e2100_ = (_e81_ < 256u);
                            if _e2100_ {
                            } else {
                                phi_5658_ = true;
                                break;
                            }
                            let _e2102_ = reduce_buf[_e81_];
                            let _e2103_ = (_e81_ + _e2097_);
                            if (_e2103_ < 256u) {
                            } else {
                                phi_5658_ = true;
                                break;
                            }
                            let _e2106_ = reduce_buf[_e2103_];
                            if _e2100_ {
                            } else {
                                phi_5658_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e2102_ + _e2106_);
                        }
                        workgroupBarrier();
                        phi_792_ = (_e2097_ >> bitcast<u32>(1i));
                    } else {
                        phi_792_ = u32();
                    }
                    let _e2111_ = phi_792_;
                    phi_791_ = _e2111_;
                    phi_5658_ = _e1975_;
                    if !(_e2098_) {
                        break;
                    }
                }
                let _e2114_ = phi_5658_;
                if _e2114_ {
                    break;
                }
                let _e2116_ = reduce_buf[0];
                let _e2117_ = bitcast<u32>(_e2116_);
                let _e2118_ = (_e2117_ & 2139095040u);
                if (_e2118_ == 2139095040u) {
                    phi_839_ = true;
                } else {
                    phi_839_ = (_e2118_ == 0u);
                }
                let _e2122_ = phi_839_;
                workgroupBarrier();
                let _e2125_ = bitcast<u32>(_e2034_);
                let _e2126_ = (_e2125_ & 2139095040u);
                if (_e2126_ == 2139095040u) {
                    phi_855_ = true;
                } else {
                    phi_855_ = (_e2126_ == 0u);
                }
                let _e2130_ = phi_855_;
                if _e2003_ {
                } else {
                    break;
                }
                reduce_buf[_e2002_] = bitcast<f32>(select(_e2125_, 0u, _e2130_));
                workgroupBarrier();
                phi_879_ = 128u;
                loop {
                    let _e2134_ = phi_879_;
                    let _e2135_ = (_e2134_ > 0u);
                    if _e2135_ {
                        if (_e81_ < _e2134_) {
                            let _e2137_ = (_e81_ < 256u);
                            if _e2137_ {
                            } else {
                                phi_5689_ = true;
                                break;
                            }
                            let _e2139_ = reduce_buf[_e81_];
                            let _e2140_ = (_e81_ + _e2134_);
                            if (_e2140_ < 256u) {
                            } else {
                                phi_5689_ = true;
                                break;
                            }
                            let _e2143_ = reduce_buf[_e2140_];
                            if _e2137_ {
                            } else {
                                phi_5689_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e2139_ + _e2143_);
                        }
                        workgroupBarrier();
                        phi_880_ = (_e2134_ >> bitcast<u32>(1i));
                    } else {
                        phi_880_ = u32();
                    }
                    let _e2148_ = phi_880_;
                    phi_879_ = _e2148_;
                    phi_5689_ = _e2114_;
                    if !(_e2135_) {
                        break;
                    }
                }
                let _e2151_ = phi_5689_;
                if _e2151_ {
                    break;
                }
                let _e2152_ = reduce_buf[0];
                let _e2153_ = bitcast<u32>(_e2152_);
                let _e2154_ = (_e2153_ & 2139095040u);
                if (_e2154_ == 2139095040u) {
                    phi_927_ = true;
                } else {
                    phi_927_ = (_e2154_ == 0u);
                }
                let _e2158_ = phi_927_;
                workgroupBarrier();
                let _e2161_ = bitcast<u32>(_e2047_);
                let _e2162_ = (_e2161_ & 2139095040u);
                if (_e2162_ == 2139095040u) {
                    phi_943_ = true;
                } else {
                    phi_943_ = (_e2162_ == 0u);
                }
                let _e2166_ = phi_943_;
                if _e2003_ {
                } else {
                    break;
                }
                reduce_buf[_e2002_] = bitcast<f32>(select(_e2161_, 0u, _e2166_));
                workgroupBarrier();
                phi_967_ = 128u;
                loop {
                    let _e2170_ = phi_967_;
                    let _e2171_ = (_e2170_ > 0u);
                    if _e2171_ {
                        if (_e81_ < _e2170_) {
                            let _e2173_ = (_e81_ < 256u);
                            if _e2173_ {
                            } else {
                                phi_5699_ = true;
                                break;
                            }
                            let _e2175_ = reduce_buf[_e81_];
                            let _e2176_ = (_e81_ + _e2170_);
                            if (_e2176_ < 256u) {
                            } else {
                                phi_5699_ = true;
                                break;
                            }
                            let _e2179_ = reduce_buf[_e2176_];
                            if _e2173_ {
                            } else {
                                phi_5699_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e2175_ + _e2179_);
                        }
                        workgroupBarrier();
                        phi_968_ = (_e2170_ >> bitcast<u32>(1i));
                    } else {
                        phi_968_ = u32();
                    }
                    let _e2184_ = phi_968_;
                    phi_967_ = _e2184_;
                    phi_5699_ = _e2151_;
                    if !(_e2171_) {
                        break;
                    }
                }
                let _e2187_ = phi_5699_;
                if _e2187_ {
                    break;
                }
                let _e2188_ = reduce_buf[0];
                let _e2189_ = bitcast<u32>(_e2188_);
                let _e2190_ = (_e2189_ & 2139095040u);
                if (_e2190_ == 2139095040u) {
                    phi_1015_ = true;
                } else {
                    phi_1015_ = (_e2190_ == 0u);
                }
                let _e2194_ = phi_1015_;
                workgroupBarrier();
                let _e2197_ = bitcast<u32>(_e2060_);
                let _e2198_ = (_e2197_ & 2139095040u);
                if (_e2198_ == 2139095040u) {
                    phi_1031_ = true;
                } else {
                    phi_1031_ = (_e2198_ == 0u);
                }
                let _e2202_ = phi_1031_;
                if _e2003_ {
                } else {
                    break;
                }
                reduce_buf[_e2002_] = bitcast<f32>(select(_e2197_, 0u, _e2202_));
                workgroupBarrier();
                phi_1055_ = 128u;
                loop {
                    let _e2206_ = phi_1055_;
                    let _e2207_ = (_e2206_ > 0u);
                    if _e2207_ {
                        if (_e81_ < _e2206_) {
                            let _e2209_ = (_e81_ < 256u);
                            if _e2209_ {
                            } else {
                                phi_5709_ = true;
                                break;
                            }
                            let _e2211_ = reduce_buf[_e81_];
                            let _e2212_ = (_e81_ + _e2206_);
                            if (_e2212_ < 256u) {
                            } else {
                                phi_5709_ = true;
                                break;
                            }
                            let _e2215_ = reduce_buf[_e2212_];
                            if _e2209_ {
                            } else {
                                phi_5709_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e2211_ + _e2215_);
                        }
                        workgroupBarrier();
                        phi_1056_ = (_e2206_ >> bitcast<u32>(1i));
                    } else {
                        phi_1056_ = u32();
                    }
                    let _e2220_ = phi_1056_;
                    phi_1055_ = _e2220_;
                    phi_5709_ = _e2187_;
                    if !(_e2207_) {
                        break;
                    }
                }
                let _e2223_ = phi_5709_;
                if _e2223_ {
                    break;
                }
                let _e2224_ = reduce_buf[0];
                let _e2225_ = bitcast<u32>(_e2224_);
                let _e2226_ = (_e2225_ & 2139095040u);
                if (_e2226_ == 2139095040u) {
                    phi_1103_ = true;
                } else {
                    phi_1103_ = (_e2226_ == 0u);
                }
                let _e2230_ = phi_1103_;
                workgroupBarrier();
                let _e2233_ = bitcast<u32>(_e2073_);
                let _e2234_ = (_e2233_ & 2139095040u);
                if (_e2234_ == 2139095040u) {
                    phi_1119_ = true;
                } else {
                    phi_1119_ = (_e2234_ == 0u);
                }
                let _e2238_ = phi_1119_;
                if _e2003_ {
                } else {
                    break;
                }
                reduce_buf[_e2002_] = bitcast<f32>(select(_e2233_, 0u, _e2238_));
                workgroupBarrier();
                phi_1143_ = 128u;
                loop {
                    let _e2242_ = phi_1143_;
                    let _e2243_ = (_e2242_ > 0u);
                    if _e2243_ {
                        if (_e81_ < _e2242_) {
                            let _e2245_ = (_e81_ < 256u);
                            if _e2245_ {
                            } else {
                                phi_5719_ = true;
                                break;
                            }
                            let _e2247_ = reduce_buf[_e81_];
                            let _e2248_ = (_e81_ + _e2242_);
                            if (_e2248_ < 256u) {
                            } else {
                                phi_5719_ = true;
                                break;
                            }
                            let _e2251_ = reduce_buf[_e2248_];
                            if _e2245_ {
                            } else {
                                phi_5719_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e2247_ + _e2251_);
                        }
                        workgroupBarrier();
                        phi_1144_ = (_e2242_ >> bitcast<u32>(1i));
                    } else {
                        phi_1144_ = u32();
                    }
                    let _e2256_ = phi_1144_;
                    phi_1143_ = _e2256_;
                    phi_5719_ = _e2223_;
                    if !(_e2243_) {
                        break;
                    }
                }
                let _e2259_ = phi_5719_;
                if _e2259_ {
                    break;
                }
                let _e2260_ = reduce_buf[0];
                let _e2261_ = bitcast<u32>(_e2260_);
                let _e2262_ = (_e2261_ & 2139095040u);
                if (_e2262_ == 2139095040u) {
                    phi_1191_ = true;
                } else {
                    phi_1191_ = (_e2262_ == 0u);
                }
                let _e2266_ = phi_1191_;
                workgroupBarrier();
                let _e2269_ = bitcast<u32>(_e2086_);
                let _e2270_ = (_e2269_ & 2139095040u);
                if (_e2270_ == 2139095040u) {
                    phi_1207_ = true;
                } else {
                    phi_1207_ = (_e2270_ == 0u);
                }
                let _e2274_ = phi_1207_;
                if _e2003_ {
                } else {
                    break;
                }
                reduce_buf[_e2002_] = bitcast<f32>(select(_e2269_, 0u, _e2274_));
                workgroupBarrier();
                phi_1231_ = 128u;
                loop {
                    let _e2278_ = phi_1231_;
                    let _e2279_ = (_e2278_ > 0u);
                    if _e2279_ {
                        if (_e81_ < _e2278_) {
                            let _e2281_ = (_e81_ < 256u);
                            if _e2281_ {
                            } else {
                                phi_5729_ = true;
                                break;
                            }
                            let _e2283_ = reduce_buf[_e81_];
                            let _e2284_ = (_e81_ + _e2278_);
                            if (_e2284_ < 256u) {
                            } else {
                                phi_5729_ = true;
                                break;
                            }
                            let _e2287_ = reduce_buf[_e2284_];
                            if _e2281_ {
                            } else {
                                phi_5729_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e2283_ + _e2287_);
                        }
                        workgroupBarrier();
                        phi_1232_ = (_e2278_ >> bitcast<u32>(1i));
                    } else {
                        phi_1232_ = u32();
                    }
                    let _e2292_ = phi_1232_;
                    phi_1231_ = _e2292_;
                    phi_5729_ = _e2259_;
                    if !(_e2279_) {
                        break;
                    }
                }
                let _e2295_ = phi_5729_;
                if _e2295_ {
                    break;
                }
                let _e2296_ = reduce_buf[0];
                let _e2297_ = bitcast<u32>(_e2296_);
                let _e2298_ = (_e2297_ & 2139095040u);
                if (_e2298_ == 2139095040u) {
                    phi_1279_ = true;
                } else {
                    phi_1279_ = (_e2298_ == 0u);
                }
                let _e2302_ = phi_1279_;
                workgroupBarrier();
                let _e2307_ = uniforms.member.member_7_;
                if ((bitcast<u32>(_e2307_) & 2139095040u) != 2139095040u) {
                    phi_1298_ = select(true, false, (_e2307_ > 0.000000000001f));
                } else {
                    phi_1298_ = true;
                }
                let _e2314_ = phi_1298_;
                let _e2315_ = select(_e2307_, 0.000000000001f, _e2314_);
                let _e2318_ = uniforms.member.member_4_;
                let _e2319_ = bitcast<u32>(_e2318_);
                let _e2320_ = (_e2319_ & 2139095040u);
                if (_e2320_ == 2139095040u) {
                    phi_1315_ = true;
                } else {
                    phi_1315_ = (_e2320_ == 0u);
                }
                let _e2324_ = phi_1315_;
                let _e2326_ = bitcast<f32>(select(_e2319_, 0u, _e2324_));
                let _e2329_ = uniforms.member.member_5_;
                let _e2330_ = bitcast<u32>(_e2329_);
                let _e2331_ = (_e2330_ & 2139095040u);
                if (_e2331_ == 2139095040u) {
                    phi_1333_ = true;
                } else {
                    phi_1333_ = (_e2331_ == 0u);
                }
                let _e2335_ = phi_1333_;
                let _e2337_ = bitcast<f32>(select(_e2330_, 0u, _e2335_));
                let _e2340_ = uniforms.member.member_6_;
                let _e2341_ = bitcast<u32>(_e2340_);
                let _e2342_ = (_e2341_ & 2139095040u);
                if (_e2342_ == 2139095040u) {
                    phi_1351_ = true;
                } else {
                    phi_1351_ = (_e2342_ == 0u);
                }
                let _e2346_ = phi_1351_;
                let _e2348_ = bitcast<f32>(select(_e2341_, 0u, _e2346_));
                let _e2349_ = (bitcast<f32>(select(_e2153_, 0u, _e2158_)) + _e2315_);
                if ((bitcast<u32>(_e2349_) & 2139095040u) != 2139095040u) {
                    let _e2353_ = (_e2349_ <= 0f);
                    if _e2353_ {
                        phi_1450_ = f32();
                    } else {
                        if (_e2349_ < 0.00001f) {
                            let _e2385_ = (select(0.000000000000000000000001f, _e2349_, (_e2349_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e2385_ != _e2385_) {
                                phi_5083_ = true;
                            } else {
                                phi_5083_ = (0.000001f >= _e2385_);
                            }
                            let _e2389_ = phi_5083_;
                            let _e2390_ = select(_e2385_, 0.000001f, _e2389_);
                            let _e2393_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2390_)));
                            let _e2397_ = (_e2393_ + (_e2393_ * (1f - (_e2390_ * _e2393_))));
                            let _e2401_ = (_e2397_ + (_e2397_ * (1f - (_e2390_ * _e2397_))));
                            let _e2407_ = bitcast<u32>(((_e2401_ + (_e2401_ * (1f - (_e2390_ * _e2401_)))) * 18446744000000000000f));
                            let _e2408_ = (_e2407_ & 2139095040u);
                            if (_e2408_ == 2139095040u) {
                                phi_1408_ = true;
                            } else {
                                phi_1408_ = (_e2408_ == 0u);
                            }
                            let _e2412_ = phi_1408_;
                            phi_1449_ = bitcast<f32>(select(_e2407_, 0u, _e2412_));
                        } else {
                            if (_e2349_ != _e2349_) {
                                phi_5023_ = true;
                            } else {
                                phi_5023_ = (0.000001f >= _e2349_);
                            }
                            let _e2358_ = phi_5023_;
                            let _e2359_ = select(_e2349_, 0.000001f, _e2358_);
                            let _e2362_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2359_)));
                            let _e2366_ = (_e2362_ + (_e2362_ * (1f - (_e2359_ * _e2362_))));
                            let _e2370_ = (_e2366_ + (_e2366_ * (1f - (_e2359_ * _e2366_))));
                            let _e2375_ = bitcast<u32>((_e2370_ + (_e2370_ * (1f - (_e2359_ * _e2370_)))));
                            let _e2376_ = (_e2375_ & 2139095040u);
                            if (_e2376_ == 2139095040u) {
                                phi_1443_ = true;
                            } else {
                                phi_1443_ = (_e2376_ == 0u);
                            }
                            let _e2380_ = phi_1443_;
                            phi_1449_ = bitcast<f32>(select(_e2375_, 0u, _e2380_));
                        }
                        let _e2416_ = phi_1449_;
                        phi_1450_ = _e2416_;
                    }
                    let _e2418_ = phi_1450_;
                    phi_1451_ = _e2418_;
                    phi_1452_ = _e2353_;
                } else {
                    phi_1451_ = f32();
                    phi_1452_ = true;
                }
                let _e2420_ = phi_1451_;
                let _e2422_ = phi_1452_;
                let _e2423_ = select(_e2420_, 0f, _e2422_);
                let _e2424_ = (bitcast<f32>(select(_e2225_, 0u, _e2230_)) + _e2315_);
                if ((bitcast<u32>(_e2424_) & 2139095040u) != 2139095040u) {
                    let _e2428_ = (_e2424_ <= 0f);
                    if _e2428_ {
                        phi_1550_ = f32();
                    } else {
                        if (_e2424_ < 0.00001f) {
                            let _e2460_ = (select(0.000000000000000000000001f, _e2424_, (_e2424_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e2460_ != _e2460_) {
                                phi_5203_ = true;
                            } else {
                                phi_5203_ = (0.000001f >= _e2460_);
                            }
                            let _e2464_ = phi_5203_;
                            let _e2465_ = select(_e2460_, 0.000001f, _e2464_);
                            let _e2468_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2465_)));
                            let _e2472_ = (_e2468_ + (_e2468_ * (1f - (_e2465_ * _e2468_))));
                            let _e2476_ = (_e2472_ + (_e2472_ * (1f - (_e2465_ * _e2472_))));
                            let _e2482_ = bitcast<u32>(((_e2476_ + (_e2476_ * (1f - (_e2465_ * _e2476_)))) * 18446744000000000000f));
                            let _e2483_ = (_e2482_ & 2139095040u);
                            if (_e2483_ == 2139095040u) {
                                phi_1508_ = true;
                            } else {
                                phi_1508_ = (_e2483_ == 0u);
                            }
                            let _e2487_ = phi_1508_;
                            phi_1549_ = bitcast<f32>(select(_e2482_, 0u, _e2487_));
                        } else {
                            if (_e2424_ != _e2424_) {
                                phi_5143_ = true;
                            } else {
                                phi_5143_ = (0.000001f >= _e2424_);
                            }
                            let _e2433_ = phi_5143_;
                            let _e2434_ = select(_e2424_, 0.000001f, _e2433_);
                            let _e2437_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2434_)));
                            let _e2441_ = (_e2437_ + (_e2437_ * (1f - (_e2434_ * _e2437_))));
                            let _e2445_ = (_e2441_ + (_e2441_ * (1f - (_e2434_ * _e2441_))));
                            let _e2450_ = bitcast<u32>((_e2445_ + (_e2445_ * (1f - (_e2434_ * _e2445_)))));
                            let _e2451_ = (_e2450_ & 2139095040u);
                            if (_e2451_ == 2139095040u) {
                                phi_1543_ = true;
                            } else {
                                phi_1543_ = (_e2451_ == 0u);
                            }
                            let _e2455_ = phi_1543_;
                            phi_1549_ = bitcast<f32>(select(_e2450_, 0u, _e2455_));
                        }
                        let _e2491_ = phi_1549_;
                        phi_1550_ = _e2491_;
                    }
                    let _e2493_ = phi_1550_;
                    phi_1551_ = _e2493_;
                    phi_1552_ = _e2428_;
                } else {
                    phi_1551_ = f32();
                    phi_1552_ = true;
                }
                let _e2495_ = phi_1551_;
                let _e2497_ = phi_1552_;
                let _e2498_ = select(_e2495_, 0f, _e2497_);
                let _e2499_ = (bitcast<f32>(select(_e2297_, 0u, _e2302_)) + _e2315_);
                if ((bitcast<u32>(_e2499_) & 2139095040u) != 2139095040u) {
                    let _e2503_ = (_e2499_ <= 0f);
                    if _e2503_ {
                        phi_1650_ = f32();
                    } else {
                        if (_e2499_ < 0.00001f) {
                            let _e2535_ = (select(0.000000000000000000000001f, _e2499_, (_e2499_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e2535_ != _e2535_) {
                                phi_5323_ = true;
                            } else {
                                phi_5323_ = (0.000001f >= _e2535_);
                            }
                            let _e2539_ = phi_5323_;
                            let _e2540_ = select(_e2535_, 0.000001f, _e2539_);
                            let _e2543_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2540_)));
                            let _e2547_ = (_e2543_ + (_e2543_ * (1f - (_e2540_ * _e2543_))));
                            let _e2551_ = (_e2547_ + (_e2547_ * (1f - (_e2540_ * _e2547_))));
                            let _e2557_ = bitcast<u32>(((_e2551_ + (_e2551_ * (1f - (_e2540_ * _e2551_)))) * 18446744000000000000f));
                            let _e2558_ = (_e2557_ & 2139095040u);
                            if (_e2558_ == 2139095040u) {
                                phi_1608_ = true;
                            } else {
                                phi_1608_ = (_e2558_ == 0u);
                            }
                            let _e2562_ = phi_1608_;
                            phi_1649_ = bitcast<f32>(select(_e2557_, 0u, _e2562_));
                        } else {
                            if (_e2499_ != _e2499_) {
                                phi_5263_ = true;
                            } else {
                                phi_5263_ = (0.000001f >= _e2499_);
                            }
                            let _e2508_ = phi_5263_;
                            let _e2509_ = select(_e2499_, 0.000001f, _e2508_);
                            let _e2512_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2509_)));
                            let _e2516_ = (_e2512_ + (_e2512_ * (1f - (_e2509_ * _e2512_))));
                            let _e2520_ = (_e2516_ + (_e2516_ * (1f - (_e2509_ * _e2516_))));
                            let _e2525_ = bitcast<u32>((_e2520_ + (_e2520_ * (1f - (_e2509_ * _e2520_)))));
                            let _e2526_ = (_e2525_ & 2139095040u);
                            if (_e2526_ == 2139095040u) {
                                phi_1643_ = true;
                            } else {
                                phi_1643_ = (_e2526_ == 0u);
                            }
                            let _e2530_ = phi_1643_;
                            phi_1649_ = bitcast<f32>(select(_e2525_, 0u, _e2530_));
                        }
                        let _e2566_ = phi_1649_;
                        phi_1650_ = _e2566_;
                    }
                    let _e2568_ = phi_1650_;
                    phi_1651_ = _e2568_;
                    phi_1652_ = _e2503_;
                } else {
                    phi_1651_ = f32();
                    phi_1652_ = true;
                }
                let _e2570_ = phi_1651_;
                let _e2572_ = phi_1652_;
                let _e2573_ = select(_e2570_, 0f, _e2572_);
                let _e2575_ = bitcast<u32>((bitcast<f32>(select(_e2117_, 0u, _e2122_)) * _e2423_));
                let _e2576_ = (_e2575_ & 2139095040u);
                if (_e2576_ == 2139095040u) {
                    phi_1668_ = true;
                } else {
                    phi_1668_ = (_e2576_ == 0u);
                }
                let _e2580_ = phi_1668_;
                let _e2582_ = bitcast<f32>(select(_e2575_, 0u, _e2580_));
                let _e2584_ = bitcast<u32>((bitcast<f32>(select(_e2189_, 0u, _e2194_)) * _e2498_));
                let _e2585_ = (_e2584_ & 2139095040u);
                if (_e2585_ == 2139095040u) {
                    phi_1685_ = true;
                } else {
                    phi_1685_ = (_e2585_ == 0u);
                }
                let _e2589_ = phi_1685_;
                let _e2591_ = bitcast<f32>(select(_e2584_, 0u, _e2589_));
                let _e2593_ = bitcast<u32>((bitcast<f32>(select(_e2261_, 0u, _e2266_)) * _e2573_));
                let _e2594_ = (_e2593_ & 2139095040u);
                if (_e2594_ == 2139095040u) {
                    phi_1702_ = true;
                } else {
                    phi_1702_ = (_e2594_ == 0u);
                }
                let _e2598_ = phi_1702_;
                let _e2600_ = bitcast<f32>(select(_e2593_, 0u, _e2598_));
                let _e2606_ = bitcast<u32>((((_e2326_ * _e2582_) + (_e2337_ * _e2591_)) + (_e2348_ * _e2600_)));
                let _e2607_ = (_e2606_ & 2139095040u);
                if (_e2607_ == 2139095040u) {
                    phi_1723_ = true;
                } else {
                    phi_1723_ = (_e2607_ == 0u);
                }
                let _e2611_ = phi_1723_;
                if (_e81_ == 0u) {
                    if (_e80_.x < _e1698_) {
                        if (_e80_.y < _e1691_) {
                            let _e2619_ = ((_e80_.y * _e1698_) + _e80_.x);
                            if (_e91_ == 0u) {
                                phi_1754_ = 0u;
                            } else {
                                let _e2621_ = (_e91_ - 1u);
                                phi_1754_ = select(_e2621_, _e2619_, (_e2619_ < _e2621_));
                            }
                            let _e2625_ = phi_1754_;
                            if (_e2625_ < _e91_) {
                            } else {
                                break;
                            }
                            band_losses.member[_e2625_] = vec4<f32>(_e2582_, _e2591_, _e2600_, bitcast<f32>(select(_e2606_, 0u, _e2611_)));
                        }
                    }
                }
                let _e2630_ = ((2f * _e2326_) * _e2423_);
                let _e2632_ = ((2f * _e2337_) * _e2498_);
                let _e2634_ = ((2f * _e2348_) * _e2573_);
                if _e2003_ {
                } else {
                    break;
                }
                let _e2637_ = buf_b[_e2002_].x;
                let _e2640_ = buf_b[_e2002_].y;
                let _e2643_ = buf_b[_e2002_].z;
                let _e2646_ = buf_b[_e2002_].w;
                if _e2003_ {
                } else {
                    break;
                }
                let _e2647_ = buf_c[_e2002_].x;
                let _e2648_ = buf_c[_e2002_].y;
                if _e2003_ {
                } else {
                    break;
                }
                buf_b[_e2002_] = vec4<f32>((_e2637_ * _e2630_), (_e2640_ * _e2630_), (_e2643_ * _e2632_), (_e2646_ * _e2632_));
                if _e2003_ {
                } else {
                    break;
                }
                buf_c[_e2002_].x = (_e2647_ * _e2634_);
                buf_c[_e2002_].y = (_e2648_ * _e2634_);
                workgroupBarrier();
                if _e1716_ {
                    phi_1836_ = type_20_(0u, _e1712_);
                    phi_1839_ = 0f;
                    phi_1841_ = 0f;
                    loop {
                        let _e2658_ = phi_1836_;
                        let _e2660_ = phi_1839_;
                        let _e2662_ = phi_1841_;
                        local_15_ = _e2662_;
                        local_16_ = _e2660_;
                        if (_e2658_.member < _e2658_.member_1_) {
                            phi_1837_ = type_20_((_e2658_.member + 1u), _e2658_.member_1_);
                            phi_1864_ = type_20_(1u, _e2658_.member);
                        } else {
                            phi_1837_ = _e2658_;
                            phi_1864_ = type_20_(0u, u32());
                        }
                        let _e2675_ = phi_1837_;
                        let _e2677_ = phi_1864_;
                        let _e2681_ = (bitcast<i32>(_e2677_.member) != 0i);
                        if _e2681_ {
                            let _e2682_ = (_e81_ + _e2677_.member_1_);
                            if (_e2682_ < _e1705_) {
                                if (_e87_ == 0u) {
                                    phi_1887_ = 0u;
                                } else {
                                    let _e2685_ = (_e87_ - 1u);
                                    phi_1887_ = select(_e2685_, _e2677_.member_1_, (_e2677_.member_1_ < _e2685_));
                                }
                                let _e2689_ = phi_1887_;
                                if (_e2689_ < _e87_) {
                                } else {
                                    phi_5739_ = true;
                                    break;
                                }
                                let _e2694_ = bezout_taps.member[_e2689_].x;
                                let _e2698_ = bezout_taps.member[_e2689_].y;
                                let _e2702_ = bezout_taps.member[_e2689_].z;
                                let _e2704_ = select(255u, _e2682_, (_e2682_ < 255u));
                                let _e2705_ = (_e2704_ < 256u);
                                if _e2705_ {
                                } else {
                                    phi_5739_ = true;
                                    break;
                                }
                                let _e2708_ = buf_b[_e2704_].x;
                                let _e2711_ = buf_b[_e2704_].y;
                                let _e2714_ = buf_b[_e2704_].z;
                                let _e2717_ = buf_b[_e2704_].w;
                                if _e2705_ {
                                } else {
                                    phi_5739_ = true;
                                    break;
                                }
                                let _e2720_ = buf_c[_e2704_].x;
                                let _e2723_ = buf_c[_e2704_].y;
                                let _e2724_ = bitcast<u32>(_e2694_);
                                let _e2725_ = (_e2724_ & 2139095040u);
                                let _e2726_ = (_e2725_ == 2139095040u);
                                if _e2726_ {
                                    phi_1972_ = true;
                                } else {
                                    phi_1972_ = (_e2725_ == 0u);
                                }
                                let _e2729_ = phi_1972_;
                                let _e2733_ = bitcast<u32>(_e2698_);
                                let _e2734_ = (_e2733_ & 2139095040u);
                                let _e2735_ = (_e2734_ == 2139095040u);
                                if _e2735_ {
                                    phi_1989_ = true;
                                } else {
                                    phi_1989_ = (_e2734_ == 0u);
                                }
                                let _e2738_ = phi_1989_;
                                let _e2743_ = bitcast<u32>(_e2702_);
                                let _e2744_ = (_e2743_ & 2139095040u);
                                let _e2745_ = (_e2744_ == 2139095040u);
                                if _e2745_ {
                                    phi_2007_ = true;
                                } else {
                                    phi_2007_ = (_e2744_ == 0u);
                                }
                                let _e2748_ = phi_2007_;
                                if _e2726_ {
                                    phi_2026_ = true;
                                } else {
                                    phi_2026_ = (_e2725_ == 0u);
                                }
                                let _e2756_ = phi_2026_;
                                if _e2735_ {
                                    phi_2043_ = true;
                                } else {
                                    phi_2043_ = (_e2734_ == 0u);
                                }
                                let _e2762_ = phi_2043_;
                                if _e2745_ {
                                    phi_2061_ = true;
                                } else {
                                    phi_2061_ = (_e2744_ == 0u);
                                }
                                let _e2769_ = phi_2061_;
                                phi_2070_ = (_e2660_ + (((bitcast<f32>(select(_e2724_, 0u, _e2756_)) * _e2711_) + (bitcast<f32>(select(_e2733_, 0u, _e2762_)) * _e2717_)) + (bitcast<f32>(select(_e2743_, 0u, _e2769_)) * _e2723_)));
                                phi_2071_ = (_e2662_ + (((bitcast<f32>(select(_e2724_, 0u, _e2729_)) * _e2708_) + (bitcast<f32>(select(_e2733_, 0u, _e2738_)) * _e2714_)) + (bitcast<f32>(select(_e2743_, 0u, _e2748_)) * _e2720_)));
                            } else {
                                phi_2070_ = _e2660_;
                                phi_2071_ = _e2662_;
                            }
                            let _e2776_ = phi_2070_;
                            let _e2778_ = phi_2071_;
                            phi_1840_ = _e2776_;
                            phi_1842_ = _e2778_;
                        } else {
                            phi_1840_ = f32();
                            phi_1842_ = f32();
                        }
                        let _e2780_ = phi_1840_;
                        let _e2782_ = phi_1842_;
                        phi_1836_ = _e2675_;
                        phi_1839_ = _e2780_;
                        phi_1841_ = _e2782_;
                        phi_5739_ = _e2295_;
                        if !(_e2681_) {
                            break;
                        }
                    }
                    let _e2785_ = phi_5739_;
                    if _e2785_ {
                        break;
                    }
                    let _e2787_ = local_15_;
                    let _e2788_ = bitcast<u32>(_e2787_);
                    let _e2789_ = (_e2788_ & 2139095040u);
                    if (_e2789_ == 2139095040u) {
                        phi_2082_ = true;
                    } else {
                        phi_2082_ = (_e2789_ == 0u);
                    }
                    let _e2793_ = phi_2082_;
                    let _e2797_ = local_16_;
                    let _e2798_ = bitcast<u32>(_e2797_);
                    let _e2799_ = (_e2798_ & 2139095040u);
                    if (_e2799_ == 2139095040u) {
                        phi_2098_ = true;
                    } else {
                        phi_2098_ = (_e2799_ == 0u);
                    }
                    let _e2803_ = phi_2098_;
                    if (_e89_ == 0u) {
                        phi_2114_ = 0u;
                    } else {
                        let _e2807_ = (_e89_ - 1u);
                        phi_2114_ = select(_e2807_, _e1719_, (_e1719_ < _e2807_));
                    }
                    let _e2811_ = phi_2114_;
                    if (_e2811_ < _e89_) {
                    } else {
                        break;
                    }
                    d_loss_d_pred.member[_e2811_].x = bitcast<f32>(select(_e2788_, 0u, _e2793_));
                    d_loss_d_pred.member[_e2811_].y = bitcast<f32>(select(_e2798_, 0u, _e2803_));
                }
            } else {
                phi_2126_ = 4u;
                phi_2129_ = 16u;
                loop {
                    let _e99_ = phi_2126_;
                    let _e101_ = phi_2129_;
                    local = _e101_;
                    local_1_ = _e101_;
                    local_2_ = _e101_;
                    local_3_ = _e101_;
                    local_4_ = _e99_;
                    local_5_ = _e99_;
                    local_8_ = _e99_;
                    local_9_ = _e99_;
                    local_12_ = _e99_;
                    if (_e101_ < _e96_) {
                        let _e103_ = (_e101_ < 256u);
                        if _e103_ {
                            phi_2143_ = (_e99_ + 1u);
                            phi_2144_ = (_e101_ << bitcast<u32>(1i));
                        } else {
                            phi_2143_ = u32();
                            phi_2144_ = u32();
                        }
                        let _e108_ = phi_2143_;
                        let _e110_ = phi_2144_;
                        phi_2127_ = _e108_;
                        phi_2130_ = _e110_;
                        phi_2145_ = _e103_;
                    } else {
                        phi_2127_ = u32();
                        phi_2130_ = u32();
                        phi_2145_ = false;
                    }
                    let _e112_ = phi_2127_;
                    let _e114_ = phi_2130_;
                    let _e116_ = phi_2145_;
                    phi_2126_ = _e112_;
                    phi_2129_ = _e114_;
                    if !(_e116_) {
                        break;
                    }
                }
                let _e120_ = uniforms.member.member;
                let _e123_ = uniforms.member.member_3_;
                let _e125_ = select(1u, _e123_, (1u < _e123_));
                if (_e125_ <= 1u) {
                    phi_2214_ = select(_e91_, 0u, (_e125_ == 0u));
                } else {
                    if (_e91_ < _e125_) {
                        phi_2213_ = 0u;
                    } else {
                        phi_2173_ = 1u;
                        phi_2176_ = _e125_;
                        loop {
                            let _e129_ = phi_2173_;
                            let _e131_ = phi_2176_;
                            local_17_ = _e129_;
                            local_18_ = _e131_;
                            let _e134_ = (_e131_ <= (_e91_ >> bitcast<u32>(1i)));
                            if _e134_ {
                                phi_2174_ = (_e129_ << bitcast<u32>(1i));
                                phi_2177_ = (_e131_ << bitcast<u32>(1i));
                            } else {
                                phi_2174_ = u32();
                                phi_2177_ = u32();
                            }
                            let _e140_ = phi_2174_;
                            let _e142_ = phi_2177_;
                            phi_2173_ = _e140_;
                            phi_2176_ = _e142_;
                            if !(_e134_) {
                                break;
                            }
                        }
                        let _e2832_ = local_17_;
                        phi_2188_ = _e2832_;
                        phi_2191_ = 0u;
                        let _e2836_ = local_18_;
                        phi_2193_ = _e2836_;
                        phi_2195_ = _e91_;
                        loop {
                            let _e145_ = phi_2188_;
                            let _e147_ = phi_2191_;
                            let _e149_ = phi_2193_;
                            let _e151_ = phi_2195_;
                            local_19_ = _e147_;
                            let _e152_ = (_e145_ > 0u);
                            if _e152_ {
                                if (_e151_ >= _e149_) {
                                    phi_2209_ = (_e147_ | _e145_);
                                    phi_2210_ = (_e151_ - _e149_);
                                } else {
                                    phi_2209_ = _e147_;
                                    phi_2210_ = _e151_;
                                }
                                let _e157_ = phi_2209_;
                                let _e159_ = phi_2210_;
                                phi_2189_ = (_e145_ >> bitcast<u32>(1i));
                                phi_2192_ = _e157_;
                                phi_2194_ = (_e149_ >> bitcast<u32>(1i));
                                phi_2196_ = _e159_;
                            } else {
                                phi_2189_ = u32();
                                phi_2192_ = u32();
                                phi_2194_ = u32();
                                phi_2196_ = u32();
                            }
                            let _e165_ = phi_2189_;
                            let _e167_ = phi_2192_;
                            let _e169_ = phi_2194_;
                            let _e171_ = phi_2196_;
                            phi_2188_ = _e165_;
                            phi_2191_ = _e167_;
                            phi_2193_ = _e169_;
                            phi_2195_ = _e171_;
                            if !(_e152_) {
                                break;
                            }
                        }
                        let _e2846_ = local_19_;
                        phi_2213_ = _e2846_;
                    }
                    let _e174_ = phi_2213_;
                    phi_2214_ = _e174_;
                }
                let _e178_ = phi_2214_;
                let _e180_ = select(1u, _e178_, (1u < _e178_));
                let _e183_ = uniforms.member.member_2_;
                let _e185_ = local;
                let _e188_ = local_1_;
                let _e192_ = local_2_;
                let _e194_ = (_e192_ >> bitcast<u32>(1i));
                if (_e87_ == 0u) {
                    phi_2239_ = 0u;
                } else {
                    let _e196_ = (_e87_ - 1u);
                    phi_2239_ = select(_e196_, _e81_, (_e81_ < _e196_));
                }
                let _e200_ = phi_2239_;
                if (_e200_ < _e87_) {
                } else {
                    break;
                }
                let _e205_ = bezout_taps.member[_e200_].x;
                let _e209_ = bezout_taps.member[_e200_].y;
                let _e213_ = bezout_taps.member[_e200_].z;
                let _e217_ = bezout_taps.member[_e200_].w;
                let _e219_ = local_3_;
                let _e220_ = (_e81_ < _e219_);
                if _e220_ {
                    let _e221_ = bitcast<u32>(_e205_);
                    let _e222_ = (_e221_ & 2139095040u);
                    if (_e222_ == 2139095040u) {
                        phi_2274_ = true;
                    } else {
                        phi_2274_ = (_e222_ == 0u);
                    }
                    let _e226_ = phi_2274_;
                    phi_2280_ = bitcast<f32>(select(_e221_, 0u, _e226_));
                } else {
                    phi_2280_ = 0f;
                }
                let _e230_ = phi_2280_;
                let _e232_ = select(255u, _e81_, (_e81_ < 255u));
                let _e233_ = (_e232_ < 256u);
                if _e233_ {
                } else {
                    break;
                }
                buf_c[_e232_].x = _e230_;
                buf_c[_e232_].y = 0f;
                let _e238_ = ((_e80_.x * select(_e188_, _e183_, (_e185_ < _e183_))) + _e81_);
                if _e220_ {
                    let _e239_ = (_e238_ < _e120_);
                    if _e239_ {
                        let _e240_ = (_e80_.y < _e125_);
                        let _e242_ = ((_e80_.y * _e120_) + _e238_);
                        if _e240_ {
                            if (_e83_ == 0u) {
                                phi_2325_ = 0u;
                            } else {
                                let _e244_ = (_e83_ - 1u);
                                phi_2325_ = select(_e244_, _e242_, (_e242_ < _e244_));
                            }
                            let _e248_ = phi_2325_;
                            if (_e248_ < _e83_) {
                            } else {
                                break;
                            }
                            let _e253_ = pred_audio.member[_e248_].x;
                            let _e257_ = pred_audio.member[_e248_].y;
                            if (_e85_ == 0u) {
                                phi_2348_ = 0u;
                            } else {
                                let _e259_ = (_e85_ - 1u);
                                phi_2348_ = select(_e259_, _e242_, (_e242_ < _e259_));
                            }
                            let _e263_ = phi_2348_;
                            if (_e263_ < _e85_) {
                            } else {
                                break;
                            }
                            let _e268_ = target_audio.member[_e263_].x;
                            let _e272_ = target_audio.member[_e263_].y;
                            let _e275_ = bitcast<u32>(((_e253_ - _e268_) * _e230_));
                            let _e276_ = (_e275_ & 2139095040u);
                            if (_e276_ == 2139095040u) {
                                phi_2373_ = true;
                            } else {
                                phi_2373_ = (_e276_ == 0u);
                            }
                            let _e280_ = phi_2373_;
                            let _e285_ = bitcast<u32>(((_e257_ - _e272_) * _e230_));
                            let _e286_ = (_e285_ & 2139095040u);
                            if (_e286_ == 2139095040u) {
                                phi_2391_ = true;
                            } else {
                                phi_2391_ = (_e286_ == 0u);
                            }
                            let _e290_ = phi_2391_;
                            let _e294_ = bitcast<u32>((_e268_ * _e230_));
                            let _e295_ = (_e294_ & 2139095040u);
                            if (_e295_ == 2139095040u) {
                                phi_2408_ = true;
                            } else {
                                phi_2408_ = (_e295_ == 0u);
                            }
                            let _e299_ = phi_2408_;
                            let _e303_ = bitcast<u32>((_e272_ * _e230_));
                            let _e304_ = (_e303_ & 2139095040u);
                            if (_e304_ == 2139095040u) {
                                phi_2425_ = true;
                            } else {
                                phi_2425_ = (_e304_ == 0u);
                            }
                            let _e308_ = phi_2425_;
                            phi_2431_ = bitcast<f32>(select(_e303_, 0u, _e308_));
                            phi_2432_ = bitcast<f32>(select(_e294_, 0u, _e299_));
                            phi_2433_ = bitcast<f32>(select(_e285_, 0u, _e290_));
                            phi_2434_ = bitcast<f32>(select(_e275_, 0u, _e280_));
                        } else {
                            phi_2431_ = f32();
                            phi_2432_ = f32();
                            phi_2433_ = f32();
                            phi_2434_ = f32();
                        }
                        let _e312_ = phi_2431_;
                        let _e314_ = phi_2432_;
                        let _e316_ = phi_2433_;
                        let _e318_ = phi_2434_;
                        phi_2436_ = _e312_;
                        phi_2437_ = _e314_;
                        phi_2438_ = _e316_;
                        phi_2439_ = _e318_;
                        phi_2440_ = _e242_;
                        phi_2441_ = _e240_;
                        phi_2445_ = select(true, false, _e240_);
                    } else {
                        phi_2436_ = f32();
                        phi_2437_ = f32();
                        phi_2438_ = f32();
                        phi_2439_ = f32();
                        phi_2440_ = u32();
                        phi_2441_ = bool();
                        phi_2445_ = false;
                    }
                    let _e321_ = phi_2436_;
                    let _e323_ = phi_2437_;
                    let _e325_ = phi_2438_;
                    let _e327_ = phi_2439_;
                    let _e329_ = phi_2440_;
                    let _e331_ = phi_2441_;
                    let _e333_ = phi_2445_;
                    phi_2446_ = _e321_;
                    phi_2447_ = _e323_;
                    phi_2448_ = _e325_;
                    phi_2449_ = _e327_;
                    phi_2450_ = _e329_;
                    phi_2451_ = _e331_;
                    phi_2454_ = select(true, false, _e239_);
                    phi_2455_ = _e333_;
                } else {
                    phi_2446_ = f32();
                    phi_2447_ = f32();
                    phi_2448_ = f32();
                    phi_2449_ = f32();
                    phi_2450_ = u32();
                    phi_2451_ = bool();
                    phi_2454_ = true;
                    phi_2455_ = false;
                }
                let _e336_ = phi_2446_;
                let _e338_ = phi_2447_;
                let _e340_ = phi_2448_;
                let _e342_ = phi_2449_;
                let _e344_ = phi_2450_;
                let _e346_ = phi_2451_;
                let _e348_ = phi_2454_;
                let _e350_ = phi_2455_;
                if _e348_ {
                    phi_2461_ = ((_e80_.y * _e120_) + _e238_);
                } else {
                    phi_2461_ = _e344_;
                }
                let _e354_ = phi_2461_;
                let _e356_ = select(_e350_, true, _e348_);
                let _e361_ = select(_e344_, _e354_, _e356_);
                if _e220_ {
                    phi_2478_ = 0u;
                    phi_2481_ = 0u;
                    phi_2483_ = _e81_;
                    loop {
                        let _e364_ = phi_2478_;
                        let _e366_ = phi_2481_;
                        let _e368_ = phi_2483_;
                        local_20_ = _e366_;
                        let _e370_ = local_4_;
                        let _e371_ = (_e364_ < _e370_);
                        if _e371_ {
                            phi_2479_ = (_e364_ + 1u);
                            phi_2482_ = ((_e366_ << bitcast<u32>(1i)) | (_e368_ & 1u));
                            phi_2484_ = (_e368_ >> bitcast<u32>(1i));
                        } else {
                            phi_2479_ = u32();
                            phi_2482_ = u32();
                            phi_2484_ = u32();
                        }
                        let _e380_ = phi_2479_;
                        let _e382_ = phi_2482_;
                        let _e384_ = phi_2484_;
                        phi_2478_ = _e380_;
                        phi_2481_ = _e382_;
                        phi_2483_ = _e384_;
                        if !(_e371_) {
                            break;
                        }
                    }
                    let _e2890_ = local_20_;
                    phi_2496_ = _e2890_;
                } else {
                    phi_2496_ = _e81_;
                }
                let _e387_ = phi_2496_;
                let _e390_ = select(255u, _e387_, (_e387_ < 255u));
                let _e391_ = (_e390_ < 256u);
                if _e391_ {
                } else {
                    break;
                }
                buf_a[_e390_] = vec4<f32>(select(_e342_, 0f, _e356_), 0f, select(_e340_, 0f, _e356_), 0f);
                if _e391_ {
                } else {
                    break;
                }
                buf_b[_e390_] = vec4<f32>(select(_e338_, 0f, _e356_), 0f, select(_e336_, 0f, _e356_), 0f);
                workgroupBarrier();
                phi_5379_ = false;
                phi_2534_ = 0u;
                loop {
                    let _e396_ = phi_5379_;
                    let _e398_ = phi_2534_;
                    let _e400_ = local_5_;
                    let _e401_ = (_e398_ < _e400_);
                    if _e401_ {
                        let _e402_ = (_e398_ & 31u);
                        let _e404_ = (1u << bitcast<u32>(_e402_));
                        if (_e81_ < _e194_) {
                            let _e410_ = (_e81_ & (_e404_ - 1u));
                            let _e415_ = (((_e81_ >> bitcast<u32>(_e402_)) << bitcast<u32>(((_e398_ + 1u) & 31u))) + _e410_);
                            let _e416_ = (_e415_ + _e404_);
                            let _e420_ = f32((_e410_ << bitcast<u32>(((31u - _e398_) & 31u))));
                            let _e421_ = (_e420_ * 0.0000000014629181f);
                            let _e422_ = bitcast<u32>(_e421_);
                            if ((_e422_ & 2139095040u) != 2139095040u) {
                                if (_e421_ == 0f) {
                                    phi_5440_ = _e396_;
                                    phi_2788_ = type_27_();
                                    phi_2789_ = true;
                                } else {
                                    let _e428_ = (bitcast<f32>((_e422_ & 2147483647u)) > 65536f);
                                    if _e428_ {
                                        phi_5441_ = _e396_;
                                        phi_2787_ = type_27_();
                                    } else {
                                        let _e433_ = (bitcast<i32>(bitcast<u32>(((_e420_ * 0.0000000009313226f) + 12582912f))) - 1262485504i);
                                        let _e434_ = f32(_e433_);
                                        let _e440_ = (((_e421_ - (_e434_ * 1.5703125f)) - (_e434_ * 0.0004837513f)) - (_e434_ * 0.0000000754979f));
                                        let _e441_ = bitcast<u32>(_e440_);
                                        if ((_e441_ & 2139095040u) != 2139095040u) {
                                            let _e446_ = (bitcast<f32>((_e441_ & 2147483647u)) > 0.85f);
                                            if _e446_ {
                                                phi_5444_ = _e396_;
                                                phi_2779_ = type_27_();
                                            } else {
                                                let _e447_ = (_e440_ * _e440_);
                                                let _e460_ = (_e440_ + ((_e440_ * _e447_) * (-0.16666667f + (_e447_ * (0.008333334f + (_e447_ * (-0.0001984127f + (_e447_ * (0.0000027557319f + (_e447_ * (-0.000000025052108f + (_e447_ * 0.00000000016059044f))))))))))));
                                                let _e473_ = (_e447_ * (-0.5f + (_e447_ * (0.041666668f + (_e447_ * (-0.0013888889f + (_e447_ * (0.000024801588f + (_e447_ * (-0.0000002755732f + (_e447_ * (0.0000000020876756f + (_e447_ * -0.000000000011470746f)))))))))))));
                                                let _e474_ = (1f + _e473_);
                                                switch (_e433_ & 3i) {
                                                    case 0: {
                                                        phi_2643_ = _e460_;
                                                        phi_2644_ = _e474_;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_2643_ = _e474_;
                                                        phi_2644_ = -(_e460_);
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_2643_ = -(_e460_);
                                                        phi_2644_ = (-1f - _e473_);
                                                        break;
                                                    }
                                                    default: {
                                                        phi_2643_ = (-1f - _e473_);
                                                        phi_2644_ = _e460_;
                                                        break;
                                                    }
                                                }
                                                let _e481_ = phi_2643_;
                                                let _e483_ = phi_2644_;
                                                let _e484_ = bitcast<u32>(_e483_);
                                                let _e485_ = bitcast<u32>(_e481_);
                                                if ((_e484_ & 2147483647u) == 0u) {
                                                    phi_2654_ = true;
                                                } else {
                                                    phi_2654_ = ((_e485_ & 2147483647u) == 0u);
                                                }
                                                let _e491_ = phi_2654_;
                                                phi_2657_ = type_32_(0i, 2i);
                                                phi_2660_ = _e481_;
                                                phi_2662_ = _e483_;
                                                phi_2664_ = _e485_;
                                                phi_2666_ = _e484_;
                                                loop {
                                                    let _e493_ = phi_2657_;
                                                    let _e495_ = phi_2660_;
                                                    let _e497_ = phi_2662_;
                                                    let _e499_ = phi_2664_;
                                                    let _e501_ = phi_2666_;
                                                    local_6_ = _e497_;
                                                    local_7_ = _e495_;
                                                    if (_e493_.member < _e493_.member_1_) {
                                                        let _e508_ = (_e493_.member + 1i);
                                                        if select(false, true, ((false == (_e508_ > _e493_.member)) != false)) {
                                                            phi_2693_ = type_33_(0u, i32());
                                                        } else {
                                                            phi_2693_ = type_33_(1u, _e508_);
                                                        }
                                                        let _e518_ = phi_2693_;
                                                        if (bitcast<i32>(_e518_.member) != 0i) {
                                                        } else {
                                                            phi_5373_ = true;
                                                            break;
                                                        }
                                                        phi_2658_ = type_32_(_e518_.member_1_, _e493_.member_1_);
                                                        phi_2707_ = type_33_(1u, _e493_.member);
                                                    } else {
                                                        phi_2658_ = _e493_;
                                                        phi_2707_ = type_33_(0u, i32());
                                                    }
                                                    let _e528_ = phi_2658_;
                                                    let _e530_ = phi_2707_;
                                                    if (bitcast<i32>(_e530_.member) != 0i) {
                                                        let _e536_ = ((_e497_ * _e497_) + (_e495_ * _e495_));
                                                        if (_e536_ > 1f) {
                                                            phi_2731_ = true;
                                                        } else {
                                                            if _e491_ {
                                                                phi_2729_ = false;
                                                            } else {
                                                                phi_2729_ = (_e536_ == 1f);
                                                            }
                                                            let _e540_ = phi_2729_;
                                                            phi_2731_ = _e540_;
                                                        }
                                                        let _e542_ = phi_2731_;
                                                        if _e542_ {
                                                            if ((_e501_ & 2147483647u) >= (_e499_ & 2147483647u)) {
                                                                let _e548_ = (_e501_ - 1u);
                                                                phi_2746_ = _e495_;
                                                                phi_2747_ = bitcast<f32>(_e548_);
                                                                phi_2748_ = _e499_;
                                                                phi_2749_ = _e548_;
                                                            } else {
                                                                let _e546_ = (_e499_ - 1u);
                                                                phi_2746_ = bitcast<f32>(_e546_);
                                                                phi_2747_ = _e497_;
                                                                phi_2748_ = _e546_;
                                                                phi_2749_ = _e501_;
                                                            }
                                                            let _e551_ = phi_2746_;
                                                            let _e553_ = phi_2747_;
                                                            let _e555_ = phi_2748_;
                                                            let _e557_ = phi_2749_;
                                                            phi_2750_ = _e551_;
                                                            phi_2751_ = _e553_;
                                                            phi_2752_ = _e555_;
                                                            phi_2753_ = _e557_;
                                                        } else {
                                                            phi_2750_ = f32();
                                                            phi_2751_ = f32();
                                                            phi_2752_ = u32();
                                                            phi_2753_ = u32();
                                                        }
                                                        let _e559_ = phi_2750_;
                                                        let _e561_ = phi_2751_;
                                                        let _e563_ = phi_2752_;
                                                        let _e565_ = phi_2753_;
                                                        phi_2661_ = _e559_;
                                                        phi_2663_ = _e561_;
                                                        phi_2665_ = _e563_;
                                                        phi_2667_ = _e565_;
                                                        phi_2755_ = _e542_;
                                                    } else {
                                                        phi_2661_ = f32();
                                                        phi_2663_ = f32();
                                                        phi_2665_ = u32();
                                                        phi_2667_ = u32();
                                                        phi_2755_ = false;
                                                    }
                                                    let _e567_ = phi_2661_;
                                                    let _e569_ = phi_2663_;
                                                    let _e571_ = phi_2665_;
                                                    let _e573_ = phi_2667_;
                                                    let _e575_ = phi_2755_;
                                                    phi_2657_ = _e528_;
                                                    phi_2660_ = _e567_;
                                                    phi_2662_ = _e569_;
                                                    phi_2664_ = _e571_;
                                                    phi_2666_ = _e573_;
                                                    phi_5373_ = _e396_;
                                                    if !(_e575_) {
                                                        break;
                                                    }
                                                }
                                                let _e578_ = phi_5373_;
                                                phi_5431_ = _e578_;
                                                if _e578_ {
                                                    break;
                                                }
                                                let _e580_ = local_6_;
                                                let _e581_ = bitcast<u32>(_e580_);
                                                let _e588_ = local_7_;
                                                let _e589_ = bitcast<u32>(_e588_);
                                                phi_5444_ = _e578_;
                                                phi_2779_ = type_27_(bitcast<f32>((_e581_ & select(4294967295u, 2147483648u, ((_e581_ & 2139095040u) == 0u)))), bitcast<f32>((_e589_ & select(4294967295u, 2147483648u, ((_e589_ & 2139095040u) == 0u)))));
                                            }
                                            let _e597_ = phi_5444_;
                                            let _e599_ = phi_2779_;
                                            phi_5443_ = _e597_;
                                            phi_2780_ = _e599_;
                                            phi_2781_ = _e446_;
                                        } else {
                                            phi_5443_ = _e396_;
                                            phi_2780_ = type_27_();
                                            phi_2781_ = true;
                                        }
                                        let _e601_ = phi_5443_;
                                        let _e603_ = phi_2780_;
                                        let _e605_ = phi_2781_;
                                        if _e605_ {
                                            phi_2786_ = type_27_(1f, 0f);
                                        } else {
                                            phi_2786_ = _e603_;
                                        }
                                        let _e607_ = phi_2786_;
                                        phi_5441_ = _e601_;
                                        phi_2787_ = _e607_;
                                    }
                                    let _e609_ = phi_5441_;
                                    let _e611_ = phi_2787_;
                                    phi_5440_ = _e609_;
                                    phi_2788_ = _e611_;
                                    phi_2789_ = _e428_;
                                }
                                let _e613_ = phi_5440_;
                                let _e615_ = phi_2788_;
                                let _e617_ = phi_2789_;
                                phi_5439_ = _e613_;
                                phi_2790_ = _e615_;
                                phi_2791_ = _e617_;
                            } else {
                                phi_5439_ = _e396_;
                                phi_2790_ = type_27_();
                                phi_2791_ = true;
                            }
                            let _e619_ = phi_5439_;
                            let _e621_ = phi_2790_;
                            let _e623_ = phi_2791_;
                            if _e623_ {
                                phi_2796_ = type_27_(1f, 0f);
                            } else {
                                phi_2796_ = _e621_;
                            }
                            let _e625_ = phi_2796_;
                            let _e628_ = -(_e625_.member_1_);
                            let _e630_ = select(255u, _e415_, (_e415_ < 255u));
                            let _e632_ = select(255u, _e416_, (_e416_ < 255u));
                            let _e633_ = (_e630_ < 256u);
                            if _e633_ {
                            } else {
                                phi_5431_ = true;
                                break;
                            }
                            let _e635_ = buf_a[_e630_];
                            let _e636_ = (_e632_ < 256u);
                            if _e636_ {
                            } else {
                                phi_5431_ = true;
                                break;
                            }
                            let _e638_ = buf_a[_e632_];
                            let _e655_ = vec4<f32>(((_e625_.member * _e638_.x) - (_e628_ * _e638_.y)), ((_e625_.member * _e638_.y) + (_e628_ * _e638_.x)), ((_e625_.member * _e638_.z) - (_e628_ * _e638_.w)), ((_e625_.member * _e638_.w) + (_e628_ * _e638_.z)));
                            if _e633_ {
                            } else {
                                phi_5431_ = true;
                                break;
                            }
                            buf_a[_e630_] = (_e635_ + _e655_);
                            if _e636_ {
                            } else {
                                phi_5431_ = true;
                                break;
                            }
                            buf_a[_e632_] = (_e635_ - _e655_);
                            if _e633_ {
                            } else {
                                phi_5431_ = true;
                                break;
                            }
                            let _e659_ = buf_b[_e630_];
                            if _e636_ {
                            } else {
                                phi_5431_ = true;
                                break;
                            }
                            let _e661_ = buf_b[_e632_];
                            let _e678_ = vec4<f32>(((_e625_.member * _e661_.x) - (_e628_ * _e661_.y)), ((_e625_.member * _e661_.y) + (_e628_ * _e661_.x)), ((_e625_.member * _e661_.z) - (_e628_ * _e661_.w)), ((_e625_.member * _e661_.w) + (_e628_ * _e661_.z)));
                            if _e633_ {
                            } else {
                                phi_5431_ = true;
                                break;
                            }
                            buf_b[_e630_] = (_e659_ + _e678_);
                            if _e636_ {
                            } else {
                                phi_5431_ = true;
                                break;
                            }
                            buf_b[_e632_] = (_e659_ - _e678_);
                            phi_5433_ = _e619_;
                        } else {
                            phi_5433_ = _e396_;
                        }
                        let _e682_ = phi_5433_;
                        workgroupBarrier();
                        phi_5432_ = _e682_;
                        phi_2535_ = (_e398_ + 1u);
                    } else {
                        phi_5432_ = _e396_;
                        phi_2535_ = u32();
                    }
                    let _e685_ = phi_5432_;
                    let _e687_ = phi_2535_;
                    phi_5379_ = _e685_;
                    phi_2534_ = _e687_;
                    phi_5431_ = _e685_;
                    if !(_e401_) {
                        break;
                    }
                }
                let _e690_ = phi_5431_;
                if _e690_ {
                    break;
                }
                if _e233_ {
                } else {
                    break;
                }
                let _e692_ = buf_a[_e232_];
                if _e233_ {
                } else {
                    break;
                }
                let _e695_ = buf_b[_e232_].x;
                let _e698_ = buf_b[_e232_].y;
                let _e701_ = buf_b[_e232_].z;
                let _e704_ = buf_b[_e232_].w;
                if _e220_ {
                    let _e705_ = bitcast<u32>(_e209_);
                    let _e706_ = (_e705_ & 2139095040u);
                    if (_e706_ == 2139095040u) {
                        phi_2986_ = true;
                    } else {
                        phi_2986_ = (_e706_ == 0u);
                    }
                    let _e710_ = phi_2986_;
                    phi_2992_ = bitcast<f32>(select(_e705_, 0u, _e710_));
                } else {
                    phi_2992_ = 0f;
                }
                let _e714_ = phi_2992_;
                if _e220_ {
                    let _e715_ = bitcast<u32>(_e213_);
                    let _e716_ = (_e715_ & 2139095040u);
                    if (_e716_ == 2139095040u) {
                        phi_3007_ = true;
                    } else {
                        phi_3007_ = (_e716_ == 0u);
                    }
                    let _e720_ = phi_3007_;
                    phi_3013_ = bitcast<f32>(select(_e715_, 0u, _e720_));
                } else {
                    phi_3013_ = 0f;
                }
                let _e724_ = phi_3013_;
                if _e220_ {
                    let _e725_ = bitcast<u32>(_e217_);
                    let _e726_ = (_e725_ & 2139095040u);
                    if (_e726_ == 2139095040u) {
                        phi_3028_ = true;
                    } else {
                        phi_3028_ = (_e726_ == 0u);
                    }
                    let _e730_ = phi_3028_;
                    phi_3034_ = bitcast<f32>(select(_e725_, 0u, _e730_));
                } else {
                    phi_3034_ = 0f;
                }
                let _e734_ = phi_3034_;
                if _e220_ {
                    let _e746_ = bitcast<u32>(((((_e692_.x * _e692_.x) + (_e692_.y * _e692_.y)) + (_e692_.z * _e692_.z)) + (_e692_.w * _e692_.w)));
                    let _e747_ = (_e746_ & 2139095040u);
                    if (_e747_ == 2139095040u) {
                        phi_3064_ = true;
                    } else {
                        phi_3064_ = (_e747_ == 0u);
                    }
                    let _e751_ = phi_3064_;
                    phi_3070_ = bitcast<f32>(select(_e746_, 0u, _e751_));
                } else {
                    phi_3070_ = 0f;
                }
                let _e755_ = phi_3070_;
                if _e220_ {
                    let _e763_ = bitcast<u32>(((((_e695_ * _e695_) + (_e698_ * _e698_)) + (_e701_ * _e701_)) + (_e704_ * _e704_)));
                    let _e764_ = (_e763_ & 2139095040u);
                    if (_e764_ == 2139095040u) {
                        phi_3092_ = true;
                    } else {
                        phi_3092_ = (_e764_ == 0u);
                    }
                    let _e768_ = phi_3092_;
                    phi_3098_ = bitcast<f32>(select(_e763_, 0u, _e768_));
                } else {
                    phi_3098_ = 0f;
                }
                let _e772_ = phi_3098_;
                let _e774_ = bitcast<u32>((_e755_ * _e714_));
                let _e775_ = (_e774_ & 2139095040u);
                if (_e775_ == 2139095040u) {
                    phi_3110_ = true;
                } else {
                    phi_3110_ = (_e775_ == 0u);
                }
                let _e779_ = phi_3110_;
                if _e233_ {
                } else {
                    break;
                }
                reduce_buf[_e232_] = bitcast<f32>(select(_e774_, 0u, _e779_));
                workgroupBarrier();
                phi_3134_ = 128u;
                loop {
                    let _e784_ = phi_3134_;
                    let _e785_ = (_e784_ > 0u);
                    if _e785_ {
                        if (_e81_ < _e784_) {
                            let _e787_ = (_e81_ < 256u);
                            if _e787_ {
                            } else {
                                phi_5447_ = true;
                                break;
                            }
                            let _e789_ = reduce_buf[_e81_];
                            let _e790_ = (_e81_ + _e784_);
                            if (_e790_ < 256u) {
                            } else {
                                phi_5447_ = true;
                                break;
                            }
                            let _e793_ = reduce_buf[_e790_];
                            if _e787_ {
                            } else {
                                phi_5447_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e789_ + _e793_);
                        }
                        workgroupBarrier();
                        phi_3135_ = (_e784_ >> bitcast<u32>(1i));
                    } else {
                        phi_3135_ = u32();
                    }
                    let _e798_ = phi_3135_;
                    phi_3134_ = _e798_;
                    phi_5447_ = _e690_;
                    if !(_e785_) {
                        break;
                    }
                }
                let _e801_ = phi_5447_;
                if _e801_ {
                    break;
                }
                let _e803_ = reduce_buf[0];
                let _e804_ = bitcast<u32>(_e803_);
                let _e805_ = (_e804_ & 2139095040u);
                if (_e805_ == 2139095040u) {
                    phi_3182_ = true;
                } else {
                    phi_3182_ = (_e805_ == 0u);
                }
                let _e809_ = phi_3182_;
                workgroupBarrier();
                let _e813_ = bitcast<u32>((_e772_ * _e714_));
                let _e814_ = (_e813_ & 2139095040u);
                if (_e814_ == 2139095040u) {
                    phi_3199_ = true;
                } else {
                    phi_3199_ = (_e814_ == 0u);
                }
                let _e818_ = phi_3199_;
                if _e233_ {
                } else {
                    break;
                }
                reduce_buf[_e232_] = bitcast<f32>(select(_e813_, 0u, _e818_));
                workgroupBarrier();
                phi_3223_ = 128u;
                loop {
                    let _e822_ = phi_3223_;
                    let _e823_ = (_e822_ > 0u);
                    if _e823_ {
                        if (_e81_ < _e822_) {
                            let _e825_ = (_e81_ < 256u);
                            if _e825_ {
                            } else {
                                phi_5474_ = true;
                                break;
                            }
                            let _e827_ = reduce_buf[_e81_];
                            let _e828_ = (_e81_ + _e822_);
                            if (_e828_ < 256u) {
                            } else {
                                phi_5474_ = true;
                                break;
                            }
                            let _e831_ = reduce_buf[_e828_];
                            if _e825_ {
                            } else {
                                phi_5474_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e827_ + _e831_);
                        }
                        workgroupBarrier();
                        phi_3224_ = (_e822_ >> bitcast<u32>(1i));
                    } else {
                        phi_3224_ = u32();
                    }
                    let _e836_ = phi_3224_;
                    phi_3223_ = _e836_;
                    phi_5474_ = _e801_;
                    if !(_e823_) {
                        break;
                    }
                }
                let _e839_ = phi_5474_;
                if _e839_ {
                    break;
                }
                let _e840_ = reduce_buf[0];
                let _e841_ = bitcast<u32>(_e840_);
                let _e842_ = (_e841_ & 2139095040u);
                if (_e842_ == 2139095040u) {
                    phi_3271_ = true;
                } else {
                    phi_3271_ = (_e842_ == 0u);
                }
                let _e846_ = phi_3271_;
                workgroupBarrier();
                let _e850_ = bitcast<u32>((_e755_ * _e724_));
                let _e851_ = (_e850_ & 2139095040u);
                if (_e851_ == 2139095040u) {
                    phi_3288_ = true;
                } else {
                    phi_3288_ = (_e851_ == 0u);
                }
                let _e855_ = phi_3288_;
                if _e233_ {
                } else {
                    break;
                }
                reduce_buf[_e232_] = bitcast<f32>(select(_e850_, 0u, _e855_));
                workgroupBarrier();
                phi_3312_ = 128u;
                loop {
                    let _e859_ = phi_3312_;
                    let _e860_ = (_e859_ > 0u);
                    if _e860_ {
                        if (_e81_ < _e859_) {
                            let _e862_ = (_e81_ < 256u);
                            if _e862_ {
                            } else {
                                phi_5484_ = true;
                                break;
                            }
                            let _e864_ = reduce_buf[_e81_];
                            let _e865_ = (_e81_ + _e859_);
                            if (_e865_ < 256u) {
                            } else {
                                phi_5484_ = true;
                                break;
                            }
                            let _e868_ = reduce_buf[_e865_];
                            if _e862_ {
                            } else {
                                phi_5484_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e864_ + _e868_);
                        }
                        workgroupBarrier();
                        phi_3313_ = (_e859_ >> bitcast<u32>(1i));
                    } else {
                        phi_3313_ = u32();
                    }
                    let _e873_ = phi_3313_;
                    phi_3312_ = _e873_;
                    phi_5484_ = _e839_;
                    if !(_e860_) {
                        break;
                    }
                }
                let _e876_ = phi_5484_;
                if _e876_ {
                    break;
                }
                let _e877_ = reduce_buf[0];
                let _e878_ = bitcast<u32>(_e877_);
                let _e879_ = (_e878_ & 2139095040u);
                if (_e879_ == 2139095040u) {
                    phi_3360_ = true;
                } else {
                    phi_3360_ = (_e879_ == 0u);
                }
                let _e883_ = phi_3360_;
                workgroupBarrier();
                let _e887_ = bitcast<u32>((_e772_ * _e724_));
                let _e888_ = (_e887_ & 2139095040u);
                if (_e888_ == 2139095040u) {
                    phi_3377_ = true;
                } else {
                    phi_3377_ = (_e888_ == 0u);
                }
                let _e892_ = phi_3377_;
                if _e233_ {
                } else {
                    break;
                }
                reduce_buf[_e232_] = bitcast<f32>(select(_e887_, 0u, _e892_));
                workgroupBarrier();
                phi_3401_ = 128u;
                loop {
                    let _e896_ = phi_3401_;
                    let _e897_ = (_e896_ > 0u);
                    if _e897_ {
                        if (_e81_ < _e896_) {
                            let _e899_ = (_e81_ < 256u);
                            if _e899_ {
                            } else {
                                phi_5494_ = true;
                                break;
                            }
                            let _e901_ = reduce_buf[_e81_];
                            let _e902_ = (_e81_ + _e896_);
                            if (_e902_ < 256u) {
                            } else {
                                phi_5494_ = true;
                                break;
                            }
                            let _e905_ = reduce_buf[_e902_];
                            if _e899_ {
                            } else {
                                phi_5494_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e901_ + _e905_);
                        }
                        workgroupBarrier();
                        phi_3402_ = (_e896_ >> bitcast<u32>(1i));
                    } else {
                        phi_3402_ = u32();
                    }
                    let _e910_ = phi_3402_;
                    phi_3401_ = _e910_;
                    phi_5494_ = _e876_;
                    if !(_e897_) {
                        break;
                    }
                }
                let _e913_ = phi_5494_;
                if _e913_ {
                    break;
                }
                let _e914_ = reduce_buf[0];
                let _e915_ = bitcast<u32>(_e914_);
                let _e916_ = (_e915_ & 2139095040u);
                if (_e916_ == 2139095040u) {
                    phi_3449_ = true;
                } else {
                    phi_3449_ = (_e916_ == 0u);
                }
                let _e920_ = phi_3449_;
                workgroupBarrier();
                let _e924_ = bitcast<u32>((_e755_ * _e734_));
                let _e925_ = (_e924_ & 2139095040u);
                if (_e925_ == 2139095040u) {
                    phi_3466_ = true;
                } else {
                    phi_3466_ = (_e925_ == 0u);
                }
                let _e929_ = phi_3466_;
                if _e233_ {
                } else {
                    break;
                }
                reduce_buf[_e232_] = bitcast<f32>(select(_e924_, 0u, _e929_));
                workgroupBarrier();
                phi_3490_ = 128u;
                loop {
                    let _e933_ = phi_3490_;
                    let _e934_ = (_e933_ > 0u);
                    if _e934_ {
                        if (_e81_ < _e933_) {
                            let _e936_ = (_e81_ < 256u);
                            if _e936_ {
                            } else {
                                phi_5504_ = true;
                                break;
                            }
                            let _e938_ = reduce_buf[_e81_];
                            let _e939_ = (_e81_ + _e933_);
                            if (_e939_ < 256u) {
                            } else {
                                phi_5504_ = true;
                                break;
                            }
                            let _e942_ = reduce_buf[_e939_];
                            if _e936_ {
                            } else {
                                phi_5504_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e938_ + _e942_);
                        }
                        workgroupBarrier();
                        phi_3491_ = (_e933_ >> bitcast<u32>(1i));
                    } else {
                        phi_3491_ = u32();
                    }
                    let _e947_ = phi_3491_;
                    phi_3490_ = _e947_;
                    phi_5504_ = _e913_;
                    if !(_e934_) {
                        break;
                    }
                }
                let _e950_ = phi_5504_;
                if _e950_ {
                    break;
                }
                let _e951_ = reduce_buf[0];
                let _e952_ = bitcast<u32>(_e951_);
                let _e953_ = (_e952_ & 2139095040u);
                if (_e953_ == 2139095040u) {
                    phi_3538_ = true;
                } else {
                    phi_3538_ = (_e953_ == 0u);
                }
                let _e957_ = phi_3538_;
                workgroupBarrier();
                let _e961_ = bitcast<u32>((_e772_ * _e734_));
                let _e962_ = (_e961_ & 2139095040u);
                if (_e962_ == 2139095040u) {
                    phi_3555_ = true;
                } else {
                    phi_3555_ = (_e962_ == 0u);
                }
                let _e966_ = phi_3555_;
                if _e233_ {
                } else {
                    break;
                }
                reduce_buf[_e232_] = bitcast<f32>(select(_e961_, 0u, _e966_));
                workgroupBarrier();
                phi_3579_ = 128u;
                loop {
                    let _e970_ = phi_3579_;
                    let _e971_ = (_e970_ > 0u);
                    if _e971_ {
                        if (_e81_ < _e970_) {
                            let _e973_ = (_e81_ < 256u);
                            if _e973_ {
                            } else {
                                phi_5514_ = true;
                                break;
                            }
                            let _e975_ = reduce_buf[_e81_];
                            let _e976_ = (_e81_ + _e970_);
                            if (_e976_ < 256u) {
                            } else {
                                phi_5514_ = true;
                                break;
                            }
                            let _e979_ = reduce_buf[_e976_];
                            if _e973_ {
                            } else {
                                phi_5514_ = true;
                                break;
                            }
                            reduce_buf[_e81_] = (_e975_ + _e979_);
                        }
                        workgroupBarrier();
                        phi_3580_ = (_e970_ >> bitcast<u32>(1i));
                    } else {
                        phi_3580_ = u32();
                    }
                    let _e984_ = phi_3580_;
                    phi_3579_ = _e984_;
                    phi_5514_ = _e950_;
                    if !(_e971_) {
                        break;
                    }
                }
                let _e987_ = phi_5514_;
                if _e987_ {
                    break;
                }
                let _e988_ = reduce_buf[0];
                let _e989_ = bitcast<u32>(_e988_);
                let _e990_ = (_e989_ & 2139095040u);
                if (_e990_ == 2139095040u) {
                    phi_3627_ = true;
                } else {
                    phi_3627_ = (_e990_ == 0u);
                }
                let _e994_ = phi_3627_;
                workgroupBarrier();
                let _e999_ = uniforms.member.member_7_;
                if ((bitcast<u32>(_e999_) & 2139095040u) != 2139095040u) {
                    phi_3646_ = select(true, false, (_e999_ > 0.000000000001f));
                } else {
                    phi_3646_ = true;
                }
                let _e1006_ = phi_3646_;
                let _e1007_ = select(_e999_, 0.000000000001f, _e1006_);
                let _e1010_ = uniforms.member.member_4_;
                let _e1011_ = bitcast<u32>(_e1010_);
                let _e1012_ = (_e1011_ & 2139095040u);
                if (_e1012_ == 2139095040u) {
                    phi_3663_ = true;
                } else {
                    phi_3663_ = (_e1012_ == 0u);
                }
                let _e1016_ = phi_3663_;
                let _e1018_ = bitcast<f32>(select(_e1011_, 0u, _e1016_));
                let _e1021_ = uniforms.member.member_5_;
                let _e1022_ = bitcast<u32>(_e1021_);
                let _e1023_ = (_e1022_ & 2139095040u);
                if (_e1023_ == 2139095040u) {
                    phi_3681_ = true;
                } else {
                    phi_3681_ = (_e1023_ == 0u);
                }
                let _e1027_ = phi_3681_;
                let _e1029_ = bitcast<f32>(select(_e1022_, 0u, _e1027_));
                let _e1032_ = uniforms.member.member_6_;
                let _e1033_ = bitcast<u32>(_e1032_);
                let _e1034_ = (_e1033_ & 2139095040u);
                if (_e1034_ == 2139095040u) {
                    phi_3699_ = true;
                } else {
                    phi_3699_ = (_e1034_ == 0u);
                }
                let _e1038_ = phi_3699_;
                let _e1040_ = bitcast<f32>(select(_e1033_, 0u, _e1038_));
                let _e1041_ = (bitcast<f32>(select(_e841_, 0u, _e846_)) + _e1007_);
                if ((bitcast<u32>(_e1041_) & 2139095040u) != 2139095040u) {
                    let _e1045_ = (_e1041_ <= 0f);
                    if _e1045_ {
                        phi_3798_ = f32();
                    } else {
                        if (_e1041_ < 0.00001f) {
                            let _e1077_ = (select(0.000000000000000000000001f, _e1041_, (_e1041_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e1077_ != _e1077_) {
                                phi_4723_ = true;
                            } else {
                                phi_4723_ = (0.000001f >= _e1077_);
                            }
                            let _e1081_ = phi_4723_;
                            let _e1082_ = select(_e1077_, 0.000001f, _e1081_);
                            let _e1085_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1082_)));
                            let _e1089_ = (_e1085_ + (_e1085_ * (1f - (_e1082_ * _e1085_))));
                            let _e1093_ = (_e1089_ + (_e1089_ * (1f - (_e1082_ * _e1089_))));
                            let _e1099_ = bitcast<u32>(((_e1093_ + (_e1093_ * (1f - (_e1082_ * _e1093_)))) * 18446744000000000000f));
                            let _e1100_ = (_e1099_ & 2139095040u);
                            if (_e1100_ == 2139095040u) {
                                phi_3756_ = true;
                            } else {
                                phi_3756_ = (_e1100_ == 0u);
                            }
                            let _e1104_ = phi_3756_;
                            phi_3797_ = bitcast<f32>(select(_e1099_, 0u, _e1104_));
                        } else {
                            if (_e1041_ != _e1041_) {
                                phi_4663_ = true;
                            } else {
                                phi_4663_ = (0.000001f >= _e1041_);
                            }
                            let _e1050_ = phi_4663_;
                            let _e1051_ = select(_e1041_, 0.000001f, _e1050_);
                            let _e1054_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1051_)));
                            let _e1058_ = (_e1054_ + (_e1054_ * (1f - (_e1051_ * _e1054_))));
                            let _e1062_ = (_e1058_ + (_e1058_ * (1f - (_e1051_ * _e1058_))));
                            let _e1067_ = bitcast<u32>((_e1062_ + (_e1062_ * (1f - (_e1051_ * _e1062_)))));
                            let _e1068_ = (_e1067_ & 2139095040u);
                            if (_e1068_ == 2139095040u) {
                                phi_3791_ = true;
                            } else {
                                phi_3791_ = (_e1068_ == 0u);
                            }
                            let _e1072_ = phi_3791_;
                            phi_3797_ = bitcast<f32>(select(_e1067_, 0u, _e1072_));
                        }
                        let _e1108_ = phi_3797_;
                        phi_3798_ = _e1108_;
                    }
                    let _e1110_ = phi_3798_;
                    phi_3799_ = _e1110_;
                    phi_3800_ = _e1045_;
                } else {
                    phi_3799_ = f32();
                    phi_3800_ = true;
                }
                let _e1112_ = phi_3799_;
                let _e1114_ = phi_3800_;
                let _e1115_ = select(_e1112_, 0f, _e1114_);
                let _e1116_ = (bitcast<f32>(select(_e915_, 0u, _e920_)) + _e1007_);
                if ((bitcast<u32>(_e1116_) & 2139095040u) != 2139095040u) {
                    let _e1120_ = (_e1116_ <= 0f);
                    if _e1120_ {
                        phi_3898_ = f32();
                    } else {
                        if (_e1116_ < 0.00001f) {
                            let _e1152_ = (select(0.000000000000000000000001f, _e1116_, (_e1116_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e1152_ != _e1152_) {
                                phi_4843_ = true;
                            } else {
                                phi_4843_ = (0.000001f >= _e1152_);
                            }
                            let _e1156_ = phi_4843_;
                            let _e1157_ = select(_e1152_, 0.000001f, _e1156_);
                            let _e1160_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1157_)));
                            let _e1164_ = (_e1160_ + (_e1160_ * (1f - (_e1157_ * _e1160_))));
                            let _e1168_ = (_e1164_ + (_e1164_ * (1f - (_e1157_ * _e1164_))));
                            let _e1174_ = bitcast<u32>(((_e1168_ + (_e1168_ * (1f - (_e1157_ * _e1168_)))) * 18446744000000000000f));
                            let _e1175_ = (_e1174_ & 2139095040u);
                            if (_e1175_ == 2139095040u) {
                                phi_3856_ = true;
                            } else {
                                phi_3856_ = (_e1175_ == 0u);
                            }
                            let _e1179_ = phi_3856_;
                            phi_3897_ = bitcast<f32>(select(_e1174_, 0u, _e1179_));
                        } else {
                            if (_e1116_ != _e1116_) {
                                phi_4783_ = true;
                            } else {
                                phi_4783_ = (0.000001f >= _e1116_);
                            }
                            let _e1125_ = phi_4783_;
                            let _e1126_ = select(_e1116_, 0.000001f, _e1125_);
                            let _e1129_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1126_)));
                            let _e1133_ = (_e1129_ + (_e1129_ * (1f - (_e1126_ * _e1129_))));
                            let _e1137_ = (_e1133_ + (_e1133_ * (1f - (_e1126_ * _e1133_))));
                            let _e1142_ = bitcast<u32>((_e1137_ + (_e1137_ * (1f - (_e1126_ * _e1137_)))));
                            let _e1143_ = (_e1142_ & 2139095040u);
                            if (_e1143_ == 2139095040u) {
                                phi_3891_ = true;
                            } else {
                                phi_3891_ = (_e1143_ == 0u);
                            }
                            let _e1147_ = phi_3891_;
                            phi_3897_ = bitcast<f32>(select(_e1142_, 0u, _e1147_));
                        }
                        let _e1183_ = phi_3897_;
                        phi_3898_ = _e1183_;
                    }
                    let _e1185_ = phi_3898_;
                    phi_3899_ = _e1185_;
                    phi_3900_ = _e1120_;
                } else {
                    phi_3899_ = f32();
                    phi_3900_ = true;
                }
                let _e1187_ = phi_3899_;
                let _e1189_ = phi_3900_;
                let _e1190_ = select(_e1187_, 0f, _e1189_);
                let _e1191_ = (bitcast<f32>(select(_e989_, 0u, _e994_)) + _e1007_);
                if ((bitcast<u32>(_e1191_) & 2139095040u) != 2139095040u) {
                    let _e1195_ = (_e1191_ <= 0f);
                    if _e1195_ {
                        phi_3998_ = f32();
                    } else {
                        if (_e1191_ < 0.00001f) {
                            let _e1227_ = (select(0.000000000000000000000001f, _e1191_, (_e1191_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e1227_ != _e1227_) {
                                phi_4963_ = true;
                            } else {
                                phi_4963_ = (0.000001f >= _e1227_);
                            }
                            let _e1231_ = phi_4963_;
                            let _e1232_ = select(_e1227_, 0.000001f, _e1231_);
                            let _e1235_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1232_)));
                            let _e1239_ = (_e1235_ + (_e1235_ * (1f - (_e1232_ * _e1235_))));
                            let _e1243_ = (_e1239_ + (_e1239_ * (1f - (_e1232_ * _e1239_))));
                            let _e1249_ = bitcast<u32>(((_e1243_ + (_e1243_ * (1f - (_e1232_ * _e1243_)))) * 18446744000000000000f));
                            let _e1250_ = (_e1249_ & 2139095040u);
                            if (_e1250_ == 2139095040u) {
                                phi_3956_ = true;
                            } else {
                                phi_3956_ = (_e1250_ == 0u);
                            }
                            let _e1254_ = phi_3956_;
                            phi_3997_ = bitcast<f32>(select(_e1249_, 0u, _e1254_));
                        } else {
                            if (_e1191_ != _e1191_) {
                                phi_4903_ = true;
                            } else {
                                phi_4903_ = (0.000001f >= _e1191_);
                            }
                            let _e1200_ = phi_4903_;
                            let _e1201_ = select(_e1191_, 0.000001f, _e1200_);
                            let _e1204_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1201_)));
                            let _e1208_ = (_e1204_ + (_e1204_ * (1f - (_e1201_ * _e1204_))));
                            let _e1212_ = (_e1208_ + (_e1208_ * (1f - (_e1201_ * _e1208_))));
                            let _e1217_ = bitcast<u32>((_e1212_ + (_e1212_ * (1f - (_e1201_ * _e1212_)))));
                            let _e1218_ = (_e1217_ & 2139095040u);
                            if (_e1218_ == 2139095040u) {
                                phi_3991_ = true;
                            } else {
                                phi_3991_ = (_e1218_ == 0u);
                            }
                            let _e1222_ = phi_3991_;
                            phi_3997_ = bitcast<f32>(select(_e1217_, 0u, _e1222_));
                        }
                        let _e1258_ = phi_3997_;
                        phi_3998_ = _e1258_;
                    }
                    let _e1260_ = phi_3998_;
                    phi_3999_ = _e1260_;
                    phi_4000_ = _e1195_;
                } else {
                    phi_3999_ = f32();
                    phi_4000_ = true;
                }
                let _e1262_ = phi_3999_;
                let _e1264_ = phi_4000_;
                let _e1265_ = select(_e1262_, 0f, _e1264_);
                let _e1267_ = bitcast<u32>((bitcast<f32>(select(_e804_, 0u, _e809_)) * _e1115_));
                let _e1268_ = (_e1267_ & 2139095040u);
                if (_e1268_ == 2139095040u) {
                    phi_4016_ = true;
                } else {
                    phi_4016_ = (_e1268_ == 0u);
                }
                let _e1272_ = phi_4016_;
                let _e1274_ = bitcast<f32>(select(_e1267_, 0u, _e1272_));
                let _e1276_ = bitcast<u32>((bitcast<f32>(select(_e878_, 0u, _e883_)) * _e1190_));
                let _e1277_ = (_e1276_ & 2139095040u);
                if (_e1277_ == 2139095040u) {
                    phi_4033_ = true;
                } else {
                    phi_4033_ = (_e1277_ == 0u);
                }
                let _e1281_ = phi_4033_;
                let _e1283_ = bitcast<f32>(select(_e1276_, 0u, _e1281_));
                let _e1285_ = bitcast<u32>((bitcast<f32>(select(_e952_, 0u, _e957_)) * _e1265_));
                let _e1286_ = (_e1285_ & 2139095040u);
                if (_e1286_ == 2139095040u) {
                    phi_4050_ = true;
                } else {
                    phi_4050_ = (_e1286_ == 0u);
                }
                let _e1290_ = phi_4050_;
                let _e1292_ = bitcast<f32>(select(_e1285_, 0u, _e1290_));
                let _e1298_ = bitcast<u32>((((_e1018_ * _e1274_) + (_e1029_ * _e1283_)) + (_e1040_ * _e1292_)));
                let _e1299_ = (_e1298_ & 2139095040u);
                if (_e1299_ == 2139095040u) {
                    phi_4071_ = true;
                } else {
                    phi_4071_ = (_e1299_ == 0u);
                }
                let _e1303_ = phi_4071_;
                if (_e81_ == 0u) {
                    if (_e80_.x < _e180_) {
                        if (_e80_.y < _e125_) {
                            let _e1311_ = ((_e80_.y * _e180_) + _e80_.x);
                            if (_e91_ == 0u) {
                                phi_4102_ = 0u;
                            } else {
                                let _e1313_ = (_e91_ - 1u);
                                phi_4102_ = select(_e1313_, _e1311_, (_e1311_ < _e1313_));
                            }
                            let _e1317_ = phi_4102_;
                            if (_e1317_ < _e91_) {
                            } else {
                                break;
                            }
                            band_losses.member[_e1317_] = vec4<f32>(_e1274_, _e1283_, _e1292_, bitcast<f32>(select(_e1298_, 0u, _e1303_)));
                        }
                    }
                }
                if _e220_ {
                    phi_4123_ = (_e692_ * (2f * ((((_e1018_ * _e1115_) * _e714_) + ((_e1029_ * _e1190_) * _e724_)) + ((_e1040_ * _e1265_) * _e734_))));
                } else {
                    phi_4123_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e1332_ = phi_4123_;
                if _e233_ {
                } else {
                    break;
                }
                buf_a[_e232_] = _e1332_;
                workgroupBarrier();
                phi_5530_ = _e987_;
                phi_4142_ = 0u;
                loop {
                    let _e1334_ = phi_5530_;
                    let _e1336_ = phi_4142_;
                    let _e1338_ = local_8_;
                    let _e1339_ = (_e1336_ < _e1338_);
                    if _e1339_ {
                        let _e1341_ = local_9_;
                        let _e1343_ = ((_e1341_ - 1u) - _e1336_);
                        let _e1344_ = (_e1343_ & 31u);
                        let _e1346_ = (1u << bitcast<u32>(_e1344_));
                        if (_e81_ < _e194_) {
                            let _e1352_ = (_e81_ & (_e1346_ - 1u));
                            let _e1357_ = (((_e81_ >> bitcast<u32>(_e1344_)) << bitcast<u32>(((_e1343_ + 1u) & 31u))) + _e1352_);
                            let _e1358_ = (_e1357_ + _e1346_);
                            let _e1362_ = f32((_e1352_ << bitcast<u32>(((31u - _e1343_) & 31u))));
                            let _e1363_ = (_e1362_ * 0.0000000014629181f);
                            let _e1364_ = bitcast<u32>(_e1363_);
                            if ((_e1364_ & 2139095040u) != 2139095040u) {
                                if (_e1363_ == 0f) {
                                    phi_5612_ = _e1334_;
                                    phi_4398_ = type_27_();
                                    phi_4399_ = true;
                                } else {
                                    let _e1370_ = (bitcast<f32>((_e1364_ & 2147483647u)) > 65536f);
                                    if _e1370_ {
                                        phi_5613_ = _e1334_;
                                        phi_4397_ = type_27_();
                                    } else {
                                        let _e1375_ = (bitcast<i32>(bitcast<u32>(((_e1362_ * 0.0000000009313226f) + 12582912f))) - 1262485504i);
                                        let _e1376_ = f32(_e1375_);
                                        let _e1382_ = (((_e1363_ - (_e1376_ * 1.5703125f)) - (_e1376_ * 0.0004837513f)) - (_e1376_ * 0.0000000754979f));
                                        let _e1383_ = bitcast<u32>(_e1382_);
                                        if ((_e1383_ & 2139095040u) != 2139095040u) {
                                            let _e1388_ = (bitcast<f32>((_e1383_ & 2147483647u)) > 0.85f);
                                            if _e1388_ {
                                                phi_5616_ = _e1334_;
                                                phi_4389_ = type_27_();
                                            } else {
                                                let _e1389_ = (_e1382_ * _e1382_);
                                                let _e1402_ = (_e1382_ + ((_e1382_ * _e1389_) * (-0.16666667f + (_e1389_ * (0.008333334f + (_e1389_ * (-0.0001984127f + (_e1389_ * (0.0000027557319f + (_e1389_ * (-0.000000025052108f + (_e1389_ * 0.00000000016059044f))))))))))));
                                                let _e1415_ = (_e1389_ * (-0.5f + (_e1389_ * (0.041666668f + (_e1389_ * (-0.0013888889f + (_e1389_ * (0.000024801588f + (_e1389_ * (-0.0000002755732f + (_e1389_ * (0.0000000020876756f + (_e1389_ * -0.000000000011470746f)))))))))))));
                                                let _e1416_ = (1f + _e1415_);
                                                switch (_e1375_ & 3i) {
                                                    case 0: {
                                                        phi_4253_ = _e1402_;
                                                        phi_4254_ = _e1416_;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_4253_ = _e1416_;
                                                        phi_4254_ = -(_e1402_);
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_4253_ = -(_e1402_);
                                                        phi_4254_ = (-1f - _e1415_);
                                                        break;
                                                    }
                                                    default: {
                                                        phi_4253_ = (-1f - _e1415_);
                                                        phi_4254_ = _e1402_;
                                                        break;
                                                    }
                                                }
                                                let _e1423_ = phi_4253_;
                                                let _e1425_ = phi_4254_;
                                                let _e1426_ = bitcast<u32>(_e1425_);
                                                let _e1427_ = bitcast<u32>(_e1423_);
                                                if ((_e1426_ & 2147483647u) == 0u) {
                                                    phi_4264_ = true;
                                                } else {
                                                    phi_4264_ = ((_e1427_ & 2147483647u) == 0u);
                                                }
                                                let _e1433_ = phi_4264_;
                                                phi_4267_ = type_32_(0i, 2i);
                                                phi_4270_ = _e1423_;
                                                phi_4272_ = _e1425_;
                                                phi_4274_ = _e1427_;
                                                phi_4276_ = _e1426_;
                                                loop {
                                                    let _e1435_ = phi_4267_;
                                                    let _e1437_ = phi_4270_;
                                                    let _e1439_ = phi_4272_;
                                                    let _e1441_ = phi_4274_;
                                                    let _e1443_ = phi_4276_;
                                                    local_10_ = _e1439_;
                                                    local_11_ = _e1437_;
                                                    if (_e1435_.member < _e1435_.member_1_) {
                                                        let _e1450_ = (_e1435_.member + 1i);
                                                        if select(false, true, ((false == (_e1450_ > _e1435_.member)) != false)) {
                                                            phi_4303_ = type_33_(0u, i32());
                                                        } else {
                                                            phi_4303_ = type_33_(1u, _e1450_);
                                                        }
                                                        let _e1460_ = phi_4303_;
                                                        if (bitcast<i32>(_e1460_.member) != 0i) {
                                                        } else {
                                                            phi_5524_ = true;
                                                            break;
                                                        }
                                                        phi_4268_ = type_32_(_e1460_.member_1_, _e1435_.member_1_);
                                                        phi_4317_ = type_33_(1u, _e1435_.member);
                                                    } else {
                                                        phi_4268_ = _e1435_;
                                                        phi_4317_ = type_33_(0u, i32());
                                                    }
                                                    let _e1470_ = phi_4268_;
                                                    let _e1472_ = phi_4317_;
                                                    if (bitcast<i32>(_e1472_.member) != 0i) {
                                                        let _e1478_ = ((_e1439_ * _e1439_) + (_e1437_ * _e1437_));
                                                        if (_e1478_ > 1f) {
                                                            phi_4341_ = true;
                                                        } else {
                                                            if _e1433_ {
                                                                phi_4339_ = false;
                                                            } else {
                                                                phi_4339_ = (_e1478_ == 1f);
                                                            }
                                                            let _e1482_ = phi_4339_;
                                                            phi_4341_ = _e1482_;
                                                        }
                                                        let _e1484_ = phi_4341_;
                                                        if _e1484_ {
                                                            if ((_e1443_ & 2147483647u) >= (_e1441_ & 2147483647u)) {
                                                                let _e1490_ = (_e1443_ - 1u);
                                                                phi_4356_ = _e1437_;
                                                                phi_4357_ = bitcast<f32>(_e1490_);
                                                                phi_4358_ = _e1441_;
                                                                phi_4359_ = _e1490_;
                                                            } else {
                                                                let _e1488_ = (_e1441_ - 1u);
                                                                phi_4356_ = bitcast<f32>(_e1488_);
                                                                phi_4357_ = _e1439_;
                                                                phi_4358_ = _e1488_;
                                                                phi_4359_ = _e1443_;
                                                            }
                                                            let _e1493_ = phi_4356_;
                                                            let _e1495_ = phi_4357_;
                                                            let _e1497_ = phi_4358_;
                                                            let _e1499_ = phi_4359_;
                                                            phi_4360_ = _e1493_;
                                                            phi_4361_ = _e1495_;
                                                            phi_4362_ = _e1497_;
                                                            phi_4363_ = _e1499_;
                                                        } else {
                                                            phi_4360_ = f32();
                                                            phi_4361_ = f32();
                                                            phi_4362_ = u32();
                                                            phi_4363_ = u32();
                                                        }
                                                        let _e1501_ = phi_4360_;
                                                        let _e1503_ = phi_4361_;
                                                        let _e1505_ = phi_4362_;
                                                        let _e1507_ = phi_4363_;
                                                        phi_4271_ = _e1501_;
                                                        phi_4273_ = _e1503_;
                                                        phi_4275_ = _e1505_;
                                                        phi_4277_ = _e1507_;
                                                        phi_4365_ = _e1484_;
                                                    } else {
                                                        phi_4271_ = f32();
                                                        phi_4273_ = f32();
                                                        phi_4275_ = u32();
                                                        phi_4277_ = u32();
                                                        phi_4365_ = false;
                                                    }
                                                    let _e1509_ = phi_4271_;
                                                    let _e1511_ = phi_4273_;
                                                    let _e1513_ = phi_4275_;
                                                    let _e1515_ = phi_4277_;
                                                    let _e1517_ = phi_4365_;
                                                    phi_4267_ = _e1470_;
                                                    phi_4270_ = _e1509_;
                                                    phi_4272_ = _e1511_;
                                                    phi_4274_ = _e1513_;
                                                    phi_4276_ = _e1515_;
                                                    phi_5524_ = _e1334_;
                                                    if !(_e1517_) {
                                                        break;
                                                    }
                                                }
                                                let _e1520_ = phi_5524_;
                                                phi_5603_ = _e1520_;
                                                if _e1520_ {
                                                    break;
                                                }
                                                let _e1522_ = local_10_;
                                                let _e1523_ = bitcast<u32>(_e1522_);
                                                let _e1530_ = local_11_;
                                                let _e1531_ = bitcast<u32>(_e1530_);
                                                phi_5616_ = _e1520_;
                                                phi_4389_ = type_27_(bitcast<f32>((_e1523_ & select(4294967295u, 2147483648u, ((_e1523_ & 2139095040u) == 0u)))), bitcast<f32>((_e1531_ & select(4294967295u, 2147483648u, ((_e1531_ & 2139095040u) == 0u)))));
                                            }
                                            let _e1539_ = phi_5616_;
                                            let _e1541_ = phi_4389_;
                                            phi_5615_ = _e1539_;
                                            phi_4390_ = _e1541_;
                                            phi_4391_ = _e1388_;
                                        } else {
                                            phi_5615_ = _e1334_;
                                            phi_4390_ = type_27_();
                                            phi_4391_ = true;
                                        }
                                        let _e1543_ = phi_5615_;
                                        let _e1545_ = phi_4390_;
                                        let _e1547_ = phi_4391_;
                                        if _e1547_ {
                                            phi_4396_ = type_27_(1f, 0f);
                                        } else {
                                            phi_4396_ = _e1545_;
                                        }
                                        let _e1549_ = phi_4396_;
                                        phi_5613_ = _e1543_;
                                        phi_4397_ = _e1549_;
                                    }
                                    let _e1551_ = phi_5613_;
                                    let _e1553_ = phi_4397_;
                                    phi_5612_ = _e1551_;
                                    phi_4398_ = _e1553_;
                                    phi_4399_ = _e1370_;
                                }
                                let _e1555_ = phi_5612_;
                                let _e1557_ = phi_4398_;
                                let _e1559_ = phi_4399_;
                                phi_5611_ = _e1555_;
                                phi_4400_ = _e1557_;
                                phi_4401_ = _e1559_;
                            } else {
                                phi_5611_ = _e1334_;
                                phi_4400_ = type_27_();
                                phi_4401_ = true;
                            }
                            let _e1561_ = phi_5611_;
                            let _e1563_ = phi_4400_;
                            let _e1565_ = phi_4401_;
                            if _e1565_ {
                                phi_4406_ = type_27_(1f, 0f);
                            } else {
                                phi_4406_ = _e1563_;
                            }
                            let _e1567_ = phi_4406_;
                            let _e1571_ = select(255u, _e1357_, (_e1357_ < 255u));
                            let _e1573_ = select(255u, _e1358_, (_e1358_ < 255u));
                            let _e1574_ = (_e1571_ < 256u);
                            if _e1574_ {
                            } else {
                                phi_5603_ = true;
                                break;
                            }
                            let _e1576_ = buf_a[_e1571_];
                            let _e1577_ = (_e1573_ < 256u);
                            if _e1577_ {
                            } else {
                                phi_5603_ = true;
                                break;
                            }
                            let _e1579_ = buf_a[_e1573_];
                            if _e1574_ {
                            } else {
                                phi_5603_ = true;
                                break;
                            }
                            buf_a[_e1571_] = (_e1576_ + _e1579_);
                            let _e1583_ = (_e1576_.x - _e1579_.x);
                            let _e1586_ = (_e1576_.y - _e1579_.y);
                            let _e1589_ = (_e1576_.z - _e1579_.z);
                            let _e1592_ = (_e1576_.w - _e1579_.w);
                            if _e1577_ {
                            } else {
                                phi_5603_ = true;
                                break;
                            }
                            buf_a[_e1573_] = vec4<f32>(((_e1567_.member * _e1583_) - (_e1567_.member_1_ * _e1586_)), ((_e1567_.member * _e1586_) + (_e1567_.member_1_ * _e1583_)), ((_e1567_.member * _e1589_) - (_e1567_.member_1_ * _e1592_)), ((_e1567_.member * _e1592_) + (_e1567_.member_1_ * _e1589_)));
                            phi_5605_ = _e1561_;
                        } else {
                            phi_5605_ = _e1334_;
                        }
                        let _e1607_ = phi_5605_;
                        workgroupBarrier();
                        phi_5604_ = _e1607_;
                        phi_4143_ = (_e1336_ + 1u);
                    } else {
                        phi_5604_ = _e1334_;
                        phi_4143_ = u32();
                    }
                    let _e1610_ = phi_5604_;
                    let _e1612_ = phi_4143_;
                    phi_5530_ = _e1610_;
                    phi_4142_ = _e1612_;
                    phi_5603_ = _e1610_;
                    if !(_e1339_) {
                        break;
                    }
                }
                let _e1615_ = phi_5603_;
                if _e1615_ {
                    break;
                }
                if select(_e346_, select(_e346_, false, _e348_), _e356_) {
                    phi_4488_ = 0u;
                    phi_4491_ = 0u;
                    phi_4493_ = _e81_;
                    loop {
                        let _e1617_ = phi_4488_;
                        let _e1619_ = phi_4491_;
                        let _e1621_ = phi_4493_;
                        local_13_ = _e1619_;
                        local_14_ = _e1619_;
                        let _e1623_ = local_12_;
                        let _e1624_ = (_e1617_ < _e1623_);
                        if _e1624_ {
                            phi_4489_ = (_e1617_ + 1u);
                            phi_4492_ = ((_e1619_ << bitcast<u32>(1i)) | (_e1621_ & 1u));
                            phi_4494_ = (_e1621_ >> bitcast<u32>(1i));
                        } else {
                            phi_4489_ = u32();
                            phi_4492_ = u32();
                            phi_4494_ = u32();
                        }
                        let _e1633_ = phi_4489_;
                        let _e1635_ = phi_4492_;
                        let _e1637_ = phi_4494_;
                        phi_4488_ = _e1633_;
                        phi_4491_ = _e1635_;
                        phi_4493_ = _e1637_;
                        if !(_e1624_) {
                            break;
                        }
                    }
                    let _e1640_ = local_13_;
                    let _e1643_ = local_14_;
                    let _e1644_ = select(255u, _e1643_, (_e1640_ < 255u));
                    if (_e1644_ < 256u) {
                    } else {
                        break;
                    }
                    let _e1648_ = buf_a[_e1644_].x;
                    let _e1651_ = buf_a[_e1644_].z;
                    if _e233_ {
                    } else {
                        break;
                    }
                    let _e1652_ = buf_c[_e232_].x;
                    let _e1654_ = bitcast<u32>((_e1648_ * _e1652_));
                    let _e1655_ = (_e1654_ & 2139095040u);
                    if (_e1655_ == 2139095040u) {
                        phi_4567_ = true;
                    } else {
                        phi_4567_ = (_e1655_ == 0u);
                    }
                    let _e1659_ = phi_4567_;
                    let _e1663_ = bitcast<u32>((_e1651_ * _e1652_));
                    let _e1664_ = (_e1663_ & 2139095040u);
                    if (_e1664_ == 2139095040u) {
                        phi_4584_ = true;
                    } else {
                        phi_4584_ = (_e1664_ == 0u);
                    }
                    let _e1668_ = phi_4584_;
                    if (_e89_ == 0u) {
                        phi_4600_ = 0u;
                    } else {
                        let _e1672_ = (_e89_ - 1u);
                        phi_4600_ = select(_e1672_, _e361_, (_e361_ < _e1672_));
                    }
                    let _e1676_ = phi_4600_;
                    if (_e1676_ < _e89_) {
                    } else {
                        break;
                    }
                    d_loss_d_pred.member[_e1676_].x = bitcast<f32>(select(_e1654_, 0u, _e1659_));
                    d_loss_d_pred.member[_e1676_].y = bitcast<f32>(select(_e1663_, 0u, _e1668_));
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn gabor_bezout_grad(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    function_();
    return;
}
