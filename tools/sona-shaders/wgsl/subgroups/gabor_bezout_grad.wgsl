struct type_7_ {
    member: array<vec2<f32>>,
}

struct type_11_ {
    member: array<vec4<f32>>,
}

struct type_22_ {
    member: u32,
    member_1_: u32,
}

struct type_27_ {
    member: f32,
    member_1_: f32,
}

struct type_29_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: f32,
    member_5_: f32,
    member_6_: f32,
    member_7_: f32,
}

struct type_30_ {
    member: type_29_,
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
var<private> sg_lid_1_: u32;
@group(0) @binding(0) 
var<uniform> uniforms: type_30_;
@group(0) @binding(1) 
var<storage> pred_audio: type_7_;
@group(0) @binding(2) 
var<storage> target_audio: type_7_;
@group(0) @binding(3) 
var<storage> bezout_taps: type_11_;
@group(0) @binding(4) 
var<storage, read_write> d_loss_d_pred: type_7_;
@group(0) @binding(5) 
var<storage, read_write> band_losses: type_11_;
var<workgroup> buf_a: array<vec4<f32>, 256>;
var<workgroup> buf_b: array<vec4<f32>, 256>;
var<workgroup> buf_c: array<vec2<f32>, 256>;
var<workgroup> reduce_buf: array<f32, 256>;

fn function_() {
    var phi_2164_: u32;
    var phi_2167_: u32;
    var phi_2181_: u32;
    var phi_2182_: u32;
    var phi_2165_: u32;
    var phi_2168_: u32;
    var phi_2183_: bool;
    var phi_2211_: u32;
    var phi_2214_: u32;
    var phi_2212_: u32;
    var phi_2215_: u32;
    var phi_2226_: u32;
    var phi_2229_: u32;
    var phi_2231_: u32;
    var phi_2233_: u32;
    var phi_2247_: u32;
    var phi_2248_: u32;
    var phi_2227_: u32;
    var phi_2230_: u32;
    var phi_2232_: u32;
    var phi_2234_: u32;
    var phi_2251_: u32;
    var phi_2252_: u32;
    var local: u32;
    var local_1_: u32;
    var local_2_: u32;
    var phi_2277_: u32;
    var local_3_: u32;
    var phi_2312_: bool;
    var phi_2318_: f32;
    var phi_2363_: u32;
    var phi_2386_: u32;
    var phi_2411_: bool;
    var phi_2429_: bool;
    var phi_2446_: bool;
    var phi_2463_: bool;
    var phi_2469_: f32;
    var phi_2470_: f32;
    var phi_2471_: f32;
    var phi_2472_: f32;
    var phi_2474_: f32;
    var phi_2475_: f32;
    var phi_2476_: f32;
    var phi_2477_: f32;
    var phi_2478_: u32;
    var phi_2479_: bool;
    var phi_2483_: bool;
    var phi_2484_: f32;
    var phi_2485_: f32;
    var phi_2486_: f32;
    var phi_2487_: f32;
    var phi_2488_: u32;
    var phi_2489_: bool;
    var phi_2492_: bool;
    var phi_2493_: bool;
    var phi_2499_: u32;
    var phi_2516_: u32;
    var phi_2519_: u32;
    var phi_2521_: u32;
    var local_4_: u32;
    var phi_2517_: u32;
    var phi_2520_: u32;
    var phi_2522_: u32;
    var phi_2534_: u32;
    var phi_5453_: bool;
    var phi_2572_: u32;
    var local_5_: u32;
    var phi_2681_: f32;
    var phi_2682_: f32;
    var phi_2692_: bool;
    var phi_2695_: type_32_;
    var phi_2698_: f32;
    var phi_2700_: f32;
    var phi_2702_: u32;
    var phi_2704_: u32;
    var phi_2731_: type_33_;
    var phi_2696_: type_32_;
    var phi_2745_: type_33_;
    var phi_2767_: bool;
    var phi_2769_: bool;
    var phi_2784_: f32;
    var phi_2785_: f32;
    var phi_2786_: u32;
    var phi_2787_: u32;
    var phi_2788_: f32;
    var phi_2789_: f32;
    var phi_2790_: u32;
    var phi_2791_: u32;
    var phi_2699_: f32;
    var phi_2701_: f32;
    var phi_2703_: u32;
    var phi_2705_: u32;
    var phi_2793_: bool;
    var phi_5447_: bool;
    var local_6_: f32;
    var local_7_: f32;
    var phi_5518_: bool;
    var phi_2817_: type_27_;
    var phi_5517_: bool;
    var phi_2818_: type_27_;
    var phi_2819_: bool;
    var phi_2824_: type_27_;
    var phi_5515_: bool;
    var phi_2825_: type_27_;
    var phi_5514_: bool;
    var phi_2826_: type_27_;
    var phi_2827_: bool;
    var phi_5513_: bool;
    var phi_2828_: type_27_;
    var phi_2829_: bool;
    var phi_2834_: type_27_;
    var phi_5507_: bool;
    var phi_5506_: bool;
    var phi_2573_: u32;
    var phi_5505_: bool;
    var phi_3024_: bool;
    var phi_3030_: f32;
    var phi_3045_: bool;
    var phi_3051_: f32;
    var phi_3066_: bool;
    var phi_3072_: f32;
    var phi_3102_: bool;
    var phi_3108_: f32;
    var phi_3130_: bool;
    var phi_3136_: f32;
    var phi_3148_: bool;
    var phi_3178_: u32;
    var phi_3179_: u32;
    var phi_5521_: bool;
    var phi_3226_: bool;
    var phi_3243_: bool;
    var phi_3273_: u32;
    var phi_3274_: u32;
    var phi_5549_: bool;
    var phi_3321_: bool;
    var phi_3338_: bool;
    var phi_3368_: u32;
    var phi_3369_: u32;
    var phi_5560_: bool;
    var phi_3416_: bool;
    var phi_3433_: bool;
    var phi_3463_: u32;
    var phi_3464_: u32;
    var phi_5571_: bool;
    var phi_3511_: bool;
    var phi_3528_: bool;
    var phi_3558_: u32;
    var phi_3559_: u32;
    var phi_5582_: bool;
    var phi_3606_: bool;
    var phi_3623_: bool;
    var phi_3653_: u32;
    var phi_3654_: u32;
    var phi_5593_: bool;
    var phi_3701_: bool;
    var phi_3720_: bool;
    var phi_3737_: bool;
    var phi_3755_: bool;
    var phi_3773_: bool;
    var phi_4737_: bool;
    var phi_3865_: bool;
    var phi_4797_: bool;
    var phi_3830_: bool;
    var phi_3871_: f32;
    var phi_3872_: f32;
    var phi_3873_: f32;
    var phi_3874_: bool;
    var phi_4857_: bool;
    var phi_3965_: bool;
    var phi_4917_: bool;
    var phi_3930_: bool;
    var phi_3971_: f32;
    var phi_3972_: f32;
    var phi_3973_: f32;
    var phi_3974_: bool;
    var phi_4977_: bool;
    var phi_4065_: bool;
    var phi_5037_: bool;
    var phi_4030_: bool;
    var phi_4071_: f32;
    var phi_4072_: f32;
    var phi_4073_: f32;
    var phi_4074_: bool;
    var phi_4090_: bool;
    var phi_4107_: bool;
    var phi_4124_: bool;
    var phi_4145_: bool;
    var phi_4176_: u32;
    var phi_4197_: vec4<f32>;
    var phi_5610_: bool;
    var phi_4216_: u32;
    var local_8_: u32;
    var local_9_: u32;
    var phi_4327_: f32;
    var phi_4328_: f32;
    var phi_4338_: bool;
    var phi_4341_: type_32_;
    var phi_4344_: f32;
    var phi_4346_: f32;
    var phi_4348_: u32;
    var phi_4350_: u32;
    var phi_4377_: type_33_;
    var phi_4342_: type_32_;
    var phi_4391_: type_33_;
    var phi_4413_: bool;
    var phi_4415_: bool;
    var phi_4430_: f32;
    var phi_4431_: f32;
    var phi_4432_: u32;
    var phi_4433_: u32;
    var phi_4434_: f32;
    var phi_4435_: f32;
    var phi_4436_: u32;
    var phi_4437_: u32;
    var phi_4345_: f32;
    var phi_4347_: f32;
    var phi_4349_: u32;
    var phi_4351_: u32;
    var phi_4439_: bool;
    var phi_5604_: bool;
    var local_10_: f32;
    var local_11_: f32;
    var phi_5696_: bool;
    var phi_4463_: type_27_;
    var phi_5695_: bool;
    var phi_4464_: type_27_;
    var phi_4465_: bool;
    var phi_4470_: type_27_;
    var phi_5693_: bool;
    var phi_4471_: type_27_;
    var phi_5692_: bool;
    var phi_4472_: type_27_;
    var phi_4473_: bool;
    var phi_5691_: bool;
    var phi_4474_: type_27_;
    var phi_4475_: bool;
    var phi_4480_: type_27_;
    var phi_5685_: bool;
    var phi_5684_: bool;
    var phi_4217_: u32;
    var phi_5683_: bool;
    var phi_4562_: u32;
    var phi_4565_: u32;
    var phi_4567_: u32;
    var local_12_: u32;
    var phi_4563_: u32;
    var phi_4566_: u32;
    var phi_4568_: u32;
    var local_13_: u32;
    var local_14_: u32;
    var phi_4641_: bool;
    var phi_4658_: bool;
    var phi_4674_: u32;
    var phi_176_: u32;
    var phi_183_: u32;
    var phi_206_: bool;
    var phi_223_: u32;
    var phi_246_: u32;
    var phi_270_: bool;
    var phi_287_: bool;
    var phi_303_: bool;
    var phi_319_: bool;
    var phi_365_: type_22_;
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
    var phi_390_: f32;
    var phi_366_: type_22_;
    var phi_413_: type_22_;
    var phi_435_: u32;
    var phi_466_: bool;
    var phi_482_: bool;
    var phi_498_: bool;
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
    var phi_570_: f32;
    var phi_571_: f32;
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
    var phi_391_: f32;
    var phi_5699_: bool;
    var phi_5756_: bool;
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
    var phi_582_: f32;
    var phi_583_: f32;
    var phi_637_: bool;
    var phi_643_: f32;
    var phi_660_: bool;
    var phi_666_: f32;
    var phi_683_: bool;
    var phi_689_: f32;
    var phi_706_: bool;
    var phi_712_: f32;
    var phi_729_: bool;
    var phi_735_: f32;
    var phi_752_: bool;
    var phi_758_: f32;
    var phi_769_: bool;
    var phi_799_: u32;
    var phi_800_: u32;
    var phi_5738_: bool;
    var phi_847_: bool;
    var phi_863_: bool;
    var phi_893_: u32;
    var phi_894_: u32;
    var phi_5770_: bool;
    var phi_941_: bool;
    var phi_957_: bool;
    var phi_987_: u32;
    var phi_988_: u32;
    var phi_5781_: bool;
    var phi_1035_: bool;
    var phi_1051_: bool;
    var phi_1081_: u32;
    var phi_1082_: u32;
    var phi_5792_: bool;
    var phi_1129_: bool;
    var phi_1145_: bool;
    var phi_1175_: u32;
    var phi_1176_: u32;
    var phi_5803_: bool;
    var phi_1223_: bool;
    var phi_1239_: bool;
    var phi_1269_: u32;
    var phi_1270_: u32;
    var phi_5814_: bool;
    var phi_1317_: bool;
    var phi_1336_: bool;
    var phi_1353_: bool;
    var phi_1371_: bool;
    var phi_1389_: bool;
    var phi_5097_: bool;
    var phi_1481_: bool;
    var phi_5157_: bool;
    var phi_1446_: bool;
    var phi_1487_: f32;
    var phi_1488_: f32;
    var phi_1489_: f32;
    var phi_1490_: bool;
    var phi_5217_: bool;
    var phi_1581_: bool;
    var phi_5277_: bool;
    var phi_1546_: bool;
    var phi_1587_: f32;
    var phi_1588_: f32;
    var phi_1589_: f32;
    var phi_1590_: bool;
    var phi_5337_: bool;
    var phi_1681_: bool;
    var phi_5397_: bool;
    var phi_1646_: bool;
    var phi_1687_: f32;
    var phi_1688_: f32;
    var phi_1689_: f32;
    var phi_1690_: bool;
    var phi_1706_: bool;
    var phi_1723_: bool;
    var phi_1740_: bool;
    var phi_1761_: bool;
    var phi_1792_: u32;
    var phi_1874_: type_22_;
    var phi_1877_: f32;
    var phi_1879_: f32;
    var phi_1875_: type_22_;
    var phi_1902_: type_22_;
    var phi_1925_: u32;
    var phi_2010_: bool;
    var phi_2027_: bool;
    var phi_2045_: bool;
    var phi_2064_: bool;
    var phi_2081_: bool;
    var phi_2099_: bool;
    var phi_2108_: f32;
    var phi_2109_: f32;
    var phi_1878_: f32;
    var phi_1880_: f32;
    var phi_5825_: bool;
    var local_15_: f32;
    var phi_2120_: bool;
    var local_16_: f32;
    var phi_2136_: bool;
    var phi_2152_: u32;
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
            let _e81_ = wg_id_1_;
            let _e82_ = lid_1_;
            let _e83_ = sg_lid_1_;
            let _e85_ = arrayLength((&pred_audio.member));
            let _e87_ = arrayLength((&target_audio.member));
            let _e89_ = arrayLength((&bezout_taps.member));
            let _e91_ = arrayLength((&d_loss_d_pred.member));
            let _e93_ = arrayLength((&band_losses.member));
            let _e98_ = uniforms.member.member_1_;
            if (_e98_ == 0u) {
                let _e1701_ = uniforms.member.member;
                let _e1704_ = uniforms.member.member_3_;
                let _e1706_ = select(1u, _e1704_, (1u < _e1704_));
                if (_e1701_ == 0u) {
                    phi_176_ = 1u;
                } else {
                    phi_176_ = (((_e1701_ - 1u) >> bitcast<u32>(8i)) + 1u);
                }
                let _e1713_ = phi_176_;
                let _e1714_ = (_e81_.x * 256u);
                if (_e1701_ > _e1714_) {
                    phi_183_ = (_e1701_ - _e1714_);
                } else {
                    phi_183_ = 0u;
                }
                let _e1718_ = phi_183_;
                let _e1720_ = select(_e1718_, 256u, (256u < _e1718_));
                let _e1723_ = uniforms.member.member_2_;
                let _e1725_ = select(1u, _e1723_, (1u < _e1723_));
                let _e1727_ = select(_e1725_, 64u, (64u < _e1725_));
                if (_e82_ < _e1720_) {
                    phi_206_ = (_e81_.y < _e1706_);
                } else {
                    phi_206_ = false;
                }
                let _e1731_ = phi_206_;
                let _e1734_ = ((_e81_.y * _e1701_) + (_e1714_ + _e82_));
                if _e1731_ {
                    if (_e85_ == 0u) {
                        phi_223_ = 0u;
                    } else {
                        let _e1740_ = (_e85_ - 1u);
                        phi_223_ = select(_e1740_, _e1734_, (_e1734_ < _e1740_));
                    }
                    let _e1744_ = phi_223_;
                    if (_e1744_ < _e85_) {
                    } else {
                        break;
                    }
                    let _e1749_ = pred_audio.member[_e1744_].x;
                    let _e1753_ = pred_audio.member[_e1744_].y;
                    if (_e87_ == 0u) {
                        phi_246_ = 0u;
                    } else {
                        let _e1755_ = (_e87_ - 1u);
                        phi_246_ = select(_e1755_, _e1734_, (_e1734_ < _e1755_));
                    }
                    let _e1759_ = phi_246_;
                    if (_e1759_ < _e87_) {
                    } else {
                        break;
                    }
                    let _e1764_ = target_audio.member[_e1759_].x;
                    let _e1768_ = target_audio.member[_e1759_].y;
                    let _e1770_ = bitcast<u32>((_e1749_ - _e1764_));
                    let _e1771_ = (_e1770_ & 2139095040u);
                    if (_e1771_ == 2139095040u) {
                        phi_270_ = true;
                    } else {
                        phi_270_ = (_e1771_ == 0u);
                    }
                    let _e1775_ = phi_270_;
                    let _e1779_ = bitcast<u32>((_e1753_ - _e1768_));
                    let _e1780_ = (_e1779_ & 2139095040u);
                    if (_e1780_ == 2139095040u) {
                        phi_287_ = true;
                    } else {
                        phi_287_ = (_e1780_ == 0u);
                    }
                    let _e1784_ = phi_287_;
                    let _e1787_ = bitcast<u32>(_e1764_);
                    let _e1788_ = (_e1787_ & 2139095040u);
                    if (_e1788_ == 2139095040u) {
                        phi_303_ = true;
                    } else {
                        phi_303_ = (_e1788_ == 0u);
                    }
                    let _e1792_ = phi_303_;
                    let _e1795_ = bitcast<u32>(_e1768_);
                    let _e1796_ = (_e1795_ & 2139095040u);
                    if (_e1796_ == 2139095040u) {
                        phi_319_ = true;
                    } else {
                        phi_319_ = (_e1796_ == 0u);
                    }
                    let _e1800_ = phi_319_;
                    let _e1805_ = select(255u, _e82_, (_e82_ < 255u));
                    if (_e1805_ < 256u) {
                    } else {
                        break;
                    }
                    buf_a[_e1805_] = vec4<f32>(bitcast<f32>(select(_e1770_, 0u, _e1775_)), bitcast<f32>(select(_e1779_, 0u, _e1784_)), bitcast<f32>(select(_e1787_, 0u, _e1792_)), bitcast<f32>(select(_e1795_, 0u, _e1800_)));
                } else {
                    let _e1736_ = select(255u, _e82_, (_e82_ < 255u));
                    if (_e1736_ < 256u) {
                    } else {
                        break;
                    }
                    buf_a[_e1736_] = vec4<f32>(0f, 0f, 0f, 0f);
                }
                workgroupBarrier();
                if _e1731_ {
                    phi_365_ = type_22_(0u, _e1727_);
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
                    phi_390_ = 0f;
                    loop {
                        let _e1810_ = phi_365_;
                        let _e1812_ = phi_368_;
                        let _e1814_ = phi_370_;
                        let _e1816_ = phi_372_;
                        let _e1818_ = phi_374_;
                        let _e1820_ = phi_376_;
                        let _e1822_ = phi_378_;
                        let _e1824_ = phi_380_;
                        let _e1826_ = phi_382_;
                        let _e1828_ = phi_384_;
                        let _e1830_ = phi_386_;
                        let _e1832_ = phi_388_;
                        let _e1834_ = phi_390_;
                        local_21_ = _e1812_;
                        local_22_ = _e1814_;
                        local_23_ = _e1816_;
                        local_24_ = _e1818_;
                        local_25_ = _e1820_;
                        local_26_ = _e1822_;
                        local_27_ = _e1824_;
                        local_28_ = _e1826_;
                        local_29_ = _e1828_;
                        local_30_ = _e1830_;
                        local_31_ = _e1832_;
                        local_32_ = _e1834_;
                        if (_e1810_.member < _e1810_.member_1_) {
                            phi_366_ = type_22_((_e1810_.member + 1u), _e1810_.member_1_);
                            phi_413_ = type_22_(1u, _e1810_.member);
                        } else {
                            phi_366_ = _e1810_;
                            phi_413_ = type_22_(0u, u32());
                        }
                        let _e1847_ = phi_366_;
                        let _e1849_ = phi_413_;
                        let _e1853_ = (bitcast<i32>(_e1849_.member) != 0i);
                        if _e1853_ {
                            if (_e1849_.member_1_ <= _e82_) {
                                if (_e89_ == 0u) {
                                    phi_435_ = 0u;
                                } else {
                                    let _e1856_ = (_e89_ - 1u);
                                    phi_435_ = select(_e1856_, _e1849_.member_1_, (_e1849_.member_1_ < _e1856_));
                                }
                                let _e1860_ = phi_435_;
                                if (_e1860_ < _e89_) {
                                } else {
                                    phi_5699_ = true;
                                    break;
                                }
                                let _e1865_ = bezout_taps.member[_e1860_].x;
                                let _e1869_ = bezout_taps.member[_e1860_].y;
                                let _e1873_ = bezout_taps.member[_e1860_].z;
                                let _e1874_ = bitcast<u32>(_e1865_);
                                let _e1875_ = (_e1874_ & 2139095040u);
                                if (_e1875_ == 2139095040u) {
                                    phi_466_ = true;
                                } else {
                                    phi_466_ = (_e1875_ == 0u);
                                }
                                let _e1879_ = phi_466_;
                                let _e1881_ = bitcast<f32>(select(_e1874_, 0u, _e1879_));
                                let _e1882_ = bitcast<u32>(_e1869_);
                                let _e1883_ = (_e1882_ & 2139095040u);
                                if (_e1883_ == 2139095040u) {
                                    phi_482_ = true;
                                } else {
                                    phi_482_ = (_e1883_ == 0u);
                                }
                                let _e1887_ = phi_482_;
                                let _e1889_ = bitcast<f32>(select(_e1882_, 0u, _e1887_));
                                let _e1890_ = bitcast<u32>(_e1873_);
                                let _e1891_ = (_e1890_ & 2139095040u);
                                if (_e1891_ == 2139095040u) {
                                    phi_498_ = true;
                                } else {
                                    phi_498_ = (_e1891_ == 0u);
                                }
                                let _e1895_ = phi_498_;
                                let _e1897_ = bitcast<f32>(select(_e1890_, 0u, _e1895_));
                                let _e1898_ = (_e82_ - _e1849_.member_1_);
                                let _e1900_ = select(255u, _e1898_, (_e1898_ < 255u));
                                if (_e1900_ < 256u) {
                                } else {
                                    phi_5699_ = true;
                                    break;
                                }
                                let _e1904_ = buf_a[_e1900_].x;
                                let _e1907_ = buf_a[_e1900_].y;
                                let _e1910_ = buf_a[_e1900_].z;
                                let _e1913_ = buf_a[_e1900_].w;
                                phi_560_ = (_e1812_ + (_e1897_ * _e1913_));
                                phi_561_ = (_e1814_ + (_e1897_ * _e1910_));
                                phi_562_ = (_e1816_ + (_e1889_ * _e1913_));
                                phi_563_ = (_e1818_ + (_e1889_ * _e1910_));
                                phi_564_ = (_e1820_ + (_e1881_ * _e1913_));
                                phi_565_ = (_e1822_ + (_e1881_ * _e1910_));
                                phi_566_ = (_e1824_ + (_e1897_ * _e1907_));
                                phi_567_ = (_e1826_ + (_e1897_ * _e1904_));
                                phi_568_ = (_e1828_ + (_e1889_ * _e1907_));
                                phi_569_ = (_e1830_ + (_e1889_ * _e1904_));
                                phi_570_ = (_e1832_ + (_e1881_ * _e1907_));
                                phi_571_ = (_e1834_ + (_e1881_ * _e1904_));
                            } else {
                                phi_560_ = _e1812_;
                                phi_561_ = _e1814_;
                                phi_562_ = _e1816_;
                                phi_563_ = _e1818_;
                                phi_564_ = _e1820_;
                                phi_565_ = _e1822_;
                                phi_566_ = _e1824_;
                                phi_567_ = _e1826_;
                                phi_568_ = _e1828_;
                                phi_569_ = _e1830_;
                                phi_570_ = _e1832_;
                                phi_571_ = _e1834_;
                            }
                            let _e1939_ = phi_560_;
                            let _e1941_ = phi_561_;
                            let _e1943_ = phi_562_;
                            let _e1945_ = phi_563_;
                            let _e1947_ = phi_564_;
                            let _e1949_ = phi_565_;
                            let _e1951_ = phi_566_;
                            let _e1953_ = phi_567_;
                            let _e1955_ = phi_568_;
                            let _e1957_ = phi_569_;
                            let _e1959_ = phi_570_;
                            let _e1961_ = phi_571_;
                            phi_369_ = _e1939_;
                            phi_371_ = _e1941_;
                            phi_373_ = _e1943_;
                            phi_375_ = _e1945_;
                            phi_377_ = _e1947_;
                            phi_379_ = _e1949_;
                            phi_381_ = _e1951_;
                            phi_383_ = _e1953_;
                            phi_385_ = _e1955_;
                            phi_387_ = _e1957_;
                            phi_389_ = _e1959_;
                            phi_391_ = _e1961_;
                        } else {
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
                            phi_391_ = f32();
                        }
                        let _e1963_ = phi_369_;
                        let _e1965_ = phi_371_;
                        let _e1967_ = phi_373_;
                        let _e1969_ = phi_375_;
                        let _e1971_ = phi_377_;
                        let _e1973_ = phi_379_;
                        let _e1975_ = phi_381_;
                        let _e1977_ = phi_383_;
                        let _e1979_ = phi_385_;
                        let _e1981_ = phi_387_;
                        let _e1983_ = phi_389_;
                        let _e1985_ = phi_391_;
                        phi_365_ = _e1847_;
                        phi_368_ = _e1963_;
                        phi_370_ = _e1965_;
                        phi_372_ = _e1967_;
                        phi_374_ = _e1969_;
                        phi_376_ = _e1971_;
                        phi_378_ = _e1973_;
                        phi_380_ = _e1975_;
                        phi_382_ = _e1977_;
                        phi_384_ = _e1979_;
                        phi_386_ = _e1981_;
                        phi_388_ = _e1983_;
                        phi_390_ = _e1985_;
                        phi_5699_ = false;
                        if !(_e1853_) {
                            break;
                        }
                    }
                    let _e1988_ = phi_5699_;
                    if _e1988_ {
                        break;
                    }
                    phi_5756_ = _e1988_;
                    let _e3164_ = local_21_;
                    phi_572_ = _e3164_;
                    let _e3167_ = local_22_;
                    phi_573_ = _e3167_;
                    let _e3170_ = local_23_;
                    phi_574_ = _e3170_;
                    let _e3173_ = local_24_;
                    phi_575_ = _e3173_;
                    let _e3176_ = local_25_;
                    phi_576_ = _e3176_;
                    let _e3179_ = local_26_;
                    phi_577_ = _e3179_;
                    let _e3182_ = local_27_;
                    phi_578_ = _e3182_;
                    let _e3185_ = local_28_;
                    phi_579_ = _e3185_;
                    let _e3188_ = local_29_;
                    phi_580_ = _e3188_;
                    let _e3191_ = local_30_;
                    phi_581_ = _e3191_;
                    let _e3194_ = local_31_;
                    phi_582_ = _e3194_;
                    let _e3197_ = local_32_;
                    phi_583_ = _e3197_;
                } else {
                    phi_5756_ = false;
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
                    phi_582_ = 0f;
                    phi_583_ = 0f;
                }
                let _e1990_ = phi_5756_;
                let _e1992_ = phi_572_;
                let _e1994_ = phi_573_;
                let _e1996_ = phi_574_;
                let _e1998_ = phi_575_;
                let _e2000_ = phi_576_;
                let _e2002_ = phi_577_;
                let _e2004_ = phi_578_;
                let _e2006_ = phi_579_;
                let _e2008_ = phi_580_;
                let _e2010_ = phi_581_;
                let _e2012_ = phi_582_;
                let _e2014_ = phi_583_;
                let _e2017_ = select(255u, _e82_, (_e82_ < 255u));
                let _e2018_ = (_e2017_ < 256u);
                if _e2018_ {
                } else {
                    break;
                }
                buf_b[_e2017_] = vec4<f32>(_e2014_, _e2012_, _e2010_, _e2008_);
                if _e2018_ {
                } else {
                    break;
                }
                buf_c[_e2017_].x = _e2006_;
                buf_c[_e2017_].y = _e2004_;
                if _e1731_ {
                    let _e2027_ = bitcast<u32>(((_e2014_ * _e2014_) + (_e2012_ * _e2012_)));
                    let _e2028_ = (_e2027_ & 2139095040u);
                    if (_e2028_ == 2139095040u) {
                        phi_637_ = true;
                    } else {
                        phi_637_ = (_e2028_ == 0u);
                    }
                    let _e2032_ = phi_637_;
                    phi_643_ = bitcast<f32>(select(_e2027_, 0u, _e2032_));
                } else {
                    phi_643_ = 0f;
                }
                let _e2036_ = phi_643_;
                if _e1731_ {
                    let _e2040_ = bitcast<u32>(((_e2002_ * _e2002_) + (_e2000_ * _e2000_)));
                    let _e2041_ = (_e2040_ & 2139095040u);
                    if (_e2041_ == 2139095040u) {
                        phi_660_ = true;
                    } else {
                        phi_660_ = (_e2041_ == 0u);
                    }
                    let _e2045_ = phi_660_;
                    phi_666_ = bitcast<f32>(select(_e2040_, 0u, _e2045_));
                } else {
                    phi_666_ = 0f;
                }
                let _e2049_ = phi_666_;
                if _e1731_ {
                    let _e2053_ = bitcast<u32>(((_e2010_ * _e2010_) + (_e2008_ * _e2008_)));
                    let _e2054_ = (_e2053_ & 2139095040u);
                    if (_e2054_ == 2139095040u) {
                        phi_683_ = true;
                    } else {
                        phi_683_ = (_e2054_ == 0u);
                    }
                    let _e2058_ = phi_683_;
                    phi_689_ = bitcast<f32>(select(_e2053_, 0u, _e2058_));
                } else {
                    phi_689_ = 0f;
                }
                let _e2062_ = phi_689_;
                if _e1731_ {
                    let _e2066_ = bitcast<u32>(((_e1998_ * _e1998_) + (_e1996_ * _e1996_)));
                    let _e2067_ = (_e2066_ & 2139095040u);
                    if (_e2067_ == 2139095040u) {
                        phi_706_ = true;
                    } else {
                        phi_706_ = (_e2067_ == 0u);
                    }
                    let _e2071_ = phi_706_;
                    phi_712_ = bitcast<f32>(select(_e2066_, 0u, _e2071_));
                } else {
                    phi_712_ = 0f;
                }
                let _e2075_ = phi_712_;
                if _e1731_ {
                    let _e2079_ = bitcast<u32>(((_e2006_ * _e2006_) + (_e2004_ * _e2004_)));
                    let _e2080_ = (_e2079_ & 2139095040u);
                    if (_e2080_ == 2139095040u) {
                        phi_729_ = true;
                    } else {
                        phi_729_ = (_e2080_ == 0u);
                    }
                    let _e2084_ = phi_729_;
                    phi_735_ = bitcast<f32>(select(_e2079_, 0u, _e2084_));
                } else {
                    phi_735_ = 0f;
                }
                let _e2088_ = phi_735_;
                if _e1731_ {
                    let _e2092_ = bitcast<u32>(((_e1994_ * _e1994_) + (_e1992_ * _e1992_)));
                    let _e2093_ = (_e2092_ & 2139095040u);
                    if (_e2093_ == 2139095040u) {
                        phi_752_ = true;
                    } else {
                        phi_752_ = (_e2093_ == 0u);
                    }
                    let _e2097_ = phi_752_;
                    phi_758_ = bitcast<f32>(select(_e2092_, 0u, _e2097_));
                } else {
                    phi_758_ = 0f;
                }
                let _e2101_ = phi_758_;
                let _e2102_ = bitcast<u32>(_e2036_);
                let _e2103_ = (_e2102_ & 2139095040u);
                if (_e2103_ == 2139095040u) {
                    phi_769_ = true;
                } else {
                    phi_769_ = (_e2103_ == 0u);
                }
                let _e2107_ = phi_769_;
                let _e993 = subgroupAdd(bitcast<f32>(select(_e2102_, 0u, _e2107_)));
                let _e2111_ = (_e83_ == 0u);
                if _e2018_ {
                } else {
                    break;
                }
                reduce_buf[_e2017_] = select(0f, _e993, _e2111_);
                workgroupBarrier();
                phi_799_ = 128u;
                loop {
                    let _e2115_ = phi_799_;
                    let _e2116_ = (_e2115_ > 0u);
                    if _e2116_ {
                        if (_e82_ < _e2115_) {
                            let _e2118_ = (_e82_ < 256u);
                            if _e2118_ {
                            } else {
                                phi_5738_ = true;
                                break;
                            }
                            let _e2120_ = reduce_buf[_e82_];
                            let _e2121_ = (_e82_ + _e2115_);
                            if (_e2121_ < 256u) {
                            } else {
                                phi_5738_ = true;
                                break;
                            }
                            let _e2124_ = reduce_buf[_e2121_];
                            if _e2118_ {
                            } else {
                                phi_5738_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e2120_ + _e2124_);
                        }
                        workgroupBarrier();
                        phi_800_ = (_e2115_ >> bitcast<u32>(1i));
                    } else {
                        phi_800_ = u32();
                    }
                    let _e2129_ = phi_800_;
                    phi_799_ = _e2129_;
                    phi_5738_ = _e1990_;
                    if !(_e2116_) {
                        break;
                    }
                }
                let _e2132_ = phi_5738_;
                if _e2132_ {
                    break;
                }
                let _e2134_ = reduce_buf[0];
                let _e2135_ = bitcast<u32>(_e2134_);
                let _e2136_ = (_e2135_ & 2139095040u);
                if (_e2136_ == 2139095040u) {
                    phi_847_ = true;
                } else {
                    phi_847_ = (_e2136_ == 0u);
                }
                let _e2140_ = phi_847_;
                workgroupBarrier();
                let _e2143_ = bitcast<u32>(_e2049_);
                let _e2144_ = (_e2143_ & 2139095040u);
                if (_e2144_ == 2139095040u) {
                    phi_863_ = true;
                } else {
                    phi_863_ = (_e2144_ == 0u);
                }
                let _e2148_ = phi_863_;
                let _e1053 = subgroupAdd(bitcast<f32>(select(_e2143_, 0u, _e2148_)));
                if _e2018_ {
                } else {
                    break;
                }
                reduce_buf[_e2017_] = select(0f, _e1053, _e2111_);
                workgroupBarrier();
                phi_893_ = 128u;
                loop {
                    let _e2154_ = phi_893_;
                    let _e2155_ = (_e2154_ > 0u);
                    if _e2155_ {
                        if (_e82_ < _e2154_) {
                            let _e2157_ = (_e82_ < 256u);
                            if _e2157_ {
                            } else {
                                phi_5770_ = true;
                                break;
                            }
                            let _e2159_ = reduce_buf[_e82_];
                            let _e2160_ = (_e82_ + _e2154_);
                            if (_e2160_ < 256u) {
                            } else {
                                phi_5770_ = true;
                                break;
                            }
                            let _e2163_ = reduce_buf[_e2160_];
                            if _e2157_ {
                            } else {
                                phi_5770_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e2159_ + _e2163_);
                        }
                        workgroupBarrier();
                        phi_894_ = (_e2154_ >> bitcast<u32>(1i));
                    } else {
                        phi_894_ = u32();
                    }
                    let _e2168_ = phi_894_;
                    phi_893_ = _e2168_;
                    phi_5770_ = _e2132_;
                    if !(_e2155_) {
                        break;
                    }
                }
                let _e2171_ = phi_5770_;
                if _e2171_ {
                    break;
                }
                let _e2172_ = reduce_buf[0];
                let _e2173_ = bitcast<u32>(_e2172_);
                let _e2174_ = (_e2173_ & 2139095040u);
                if (_e2174_ == 2139095040u) {
                    phi_941_ = true;
                } else {
                    phi_941_ = (_e2174_ == 0u);
                }
                let _e2178_ = phi_941_;
                workgroupBarrier();
                let _e2181_ = bitcast<u32>(_e2062_);
                let _e2182_ = (_e2181_ & 2139095040u);
                if (_e2182_ == 2139095040u) {
                    phi_957_ = true;
                } else {
                    phi_957_ = (_e2182_ == 0u);
                }
                let _e2186_ = phi_957_;
                let _e1111 = subgroupAdd(bitcast<f32>(select(_e2181_, 0u, _e2186_)));
                if _e2018_ {
                } else {
                    break;
                }
                reduce_buf[_e2017_] = select(0f, _e1111, _e2111_);
                workgroupBarrier();
                phi_987_ = 128u;
                loop {
                    let _e2192_ = phi_987_;
                    let _e2193_ = (_e2192_ > 0u);
                    if _e2193_ {
                        if (_e82_ < _e2192_) {
                            let _e2195_ = (_e82_ < 256u);
                            if _e2195_ {
                            } else {
                                phi_5781_ = true;
                                break;
                            }
                            let _e2197_ = reduce_buf[_e82_];
                            let _e2198_ = (_e82_ + _e2192_);
                            if (_e2198_ < 256u) {
                            } else {
                                phi_5781_ = true;
                                break;
                            }
                            let _e2201_ = reduce_buf[_e2198_];
                            if _e2195_ {
                            } else {
                                phi_5781_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e2197_ + _e2201_);
                        }
                        workgroupBarrier();
                        phi_988_ = (_e2192_ >> bitcast<u32>(1i));
                    } else {
                        phi_988_ = u32();
                    }
                    let _e2206_ = phi_988_;
                    phi_987_ = _e2206_;
                    phi_5781_ = _e2171_;
                    if !(_e2193_) {
                        break;
                    }
                }
                let _e2209_ = phi_5781_;
                if _e2209_ {
                    break;
                }
                let _e2210_ = reduce_buf[0];
                let _e2211_ = bitcast<u32>(_e2210_);
                let _e2212_ = (_e2211_ & 2139095040u);
                if (_e2212_ == 2139095040u) {
                    phi_1035_ = true;
                } else {
                    phi_1035_ = (_e2212_ == 0u);
                }
                let _e2216_ = phi_1035_;
                workgroupBarrier();
                let _e2219_ = bitcast<u32>(_e2075_);
                let _e2220_ = (_e2219_ & 2139095040u);
                if (_e2220_ == 2139095040u) {
                    phi_1051_ = true;
                } else {
                    phi_1051_ = (_e2220_ == 0u);
                }
                let _e2224_ = phi_1051_;
                let _e1169 = subgroupAdd(bitcast<f32>(select(_e2219_, 0u, _e2224_)));
                if _e2018_ {
                } else {
                    break;
                }
                reduce_buf[_e2017_] = select(0f, _e1169, _e2111_);
                workgroupBarrier();
                phi_1081_ = 128u;
                loop {
                    let _e2230_ = phi_1081_;
                    let _e2231_ = (_e2230_ > 0u);
                    if _e2231_ {
                        if (_e82_ < _e2230_) {
                            let _e2233_ = (_e82_ < 256u);
                            if _e2233_ {
                            } else {
                                phi_5792_ = true;
                                break;
                            }
                            let _e2235_ = reduce_buf[_e82_];
                            let _e2236_ = (_e82_ + _e2230_);
                            if (_e2236_ < 256u) {
                            } else {
                                phi_5792_ = true;
                                break;
                            }
                            let _e2239_ = reduce_buf[_e2236_];
                            if _e2233_ {
                            } else {
                                phi_5792_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e2235_ + _e2239_);
                        }
                        workgroupBarrier();
                        phi_1082_ = (_e2230_ >> bitcast<u32>(1i));
                    } else {
                        phi_1082_ = u32();
                    }
                    let _e2244_ = phi_1082_;
                    phi_1081_ = _e2244_;
                    phi_5792_ = _e2209_;
                    if !(_e2231_) {
                        break;
                    }
                }
                let _e2247_ = phi_5792_;
                if _e2247_ {
                    break;
                }
                let _e2248_ = reduce_buf[0];
                let _e2249_ = bitcast<u32>(_e2248_);
                let _e2250_ = (_e2249_ & 2139095040u);
                if (_e2250_ == 2139095040u) {
                    phi_1129_ = true;
                } else {
                    phi_1129_ = (_e2250_ == 0u);
                }
                let _e2254_ = phi_1129_;
                workgroupBarrier();
                let _e2257_ = bitcast<u32>(_e2088_);
                let _e2258_ = (_e2257_ & 2139095040u);
                if (_e2258_ == 2139095040u) {
                    phi_1145_ = true;
                } else {
                    phi_1145_ = (_e2258_ == 0u);
                }
                let _e2262_ = phi_1145_;
                let _e1227 = subgroupAdd(bitcast<f32>(select(_e2257_, 0u, _e2262_)));
                if _e2018_ {
                } else {
                    break;
                }
                reduce_buf[_e2017_] = select(0f, _e1227, _e2111_);
                workgroupBarrier();
                phi_1175_ = 128u;
                loop {
                    let _e2268_ = phi_1175_;
                    let _e2269_ = (_e2268_ > 0u);
                    if _e2269_ {
                        if (_e82_ < _e2268_) {
                            let _e2271_ = (_e82_ < 256u);
                            if _e2271_ {
                            } else {
                                phi_5803_ = true;
                                break;
                            }
                            let _e2273_ = reduce_buf[_e82_];
                            let _e2274_ = (_e82_ + _e2268_);
                            if (_e2274_ < 256u) {
                            } else {
                                phi_5803_ = true;
                                break;
                            }
                            let _e2277_ = reduce_buf[_e2274_];
                            if _e2271_ {
                            } else {
                                phi_5803_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e2273_ + _e2277_);
                        }
                        workgroupBarrier();
                        phi_1176_ = (_e2268_ >> bitcast<u32>(1i));
                    } else {
                        phi_1176_ = u32();
                    }
                    let _e2282_ = phi_1176_;
                    phi_1175_ = _e2282_;
                    phi_5803_ = _e2247_;
                    if !(_e2269_) {
                        break;
                    }
                }
                let _e2285_ = phi_5803_;
                if _e2285_ {
                    break;
                }
                let _e2286_ = reduce_buf[0];
                let _e2287_ = bitcast<u32>(_e2286_);
                let _e2288_ = (_e2287_ & 2139095040u);
                if (_e2288_ == 2139095040u) {
                    phi_1223_ = true;
                } else {
                    phi_1223_ = (_e2288_ == 0u);
                }
                let _e2292_ = phi_1223_;
                workgroupBarrier();
                let _e2295_ = bitcast<u32>(_e2101_);
                let _e2296_ = (_e2295_ & 2139095040u);
                if (_e2296_ == 2139095040u) {
                    phi_1239_ = true;
                } else {
                    phi_1239_ = (_e2296_ == 0u);
                }
                let _e2300_ = phi_1239_;
                let _e1285 = subgroupAdd(bitcast<f32>(select(_e2295_, 0u, _e2300_)));
                if _e2018_ {
                } else {
                    break;
                }
                reduce_buf[_e2017_] = select(0f, _e1285, _e2111_);
                workgroupBarrier();
                phi_1269_ = 128u;
                loop {
                    let _e2306_ = phi_1269_;
                    let _e2307_ = (_e2306_ > 0u);
                    if _e2307_ {
                        if (_e82_ < _e2306_) {
                            let _e2309_ = (_e82_ < 256u);
                            if _e2309_ {
                            } else {
                                phi_5814_ = true;
                                break;
                            }
                            let _e2311_ = reduce_buf[_e82_];
                            let _e2312_ = (_e82_ + _e2306_);
                            if (_e2312_ < 256u) {
                            } else {
                                phi_5814_ = true;
                                break;
                            }
                            let _e2315_ = reduce_buf[_e2312_];
                            if _e2309_ {
                            } else {
                                phi_5814_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e2311_ + _e2315_);
                        }
                        workgroupBarrier();
                        phi_1270_ = (_e2306_ >> bitcast<u32>(1i));
                    } else {
                        phi_1270_ = u32();
                    }
                    let _e2320_ = phi_1270_;
                    phi_1269_ = _e2320_;
                    phi_5814_ = _e2285_;
                    if !(_e2307_) {
                        break;
                    }
                }
                let _e2323_ = phi_5814_;
                if _e2323_ {
                    break;
                }
                let _e2324_ = reduce_buf[0];
                let _e2325_ = bitcast<u32>(_e2324_);
                let _e2326_ = (_e2325_ & 2139095040u);
                if (_e2326_ == 2139095040u) {
                    phi_1317_ = true;
                } else {
                    phi_1317_ = (_e2326_ == 0u);
                }
                let _e2330_ = phi_1317_;
                workgroupBarrier();
                let _e2335_ = uniforms.member.member_7_;
                if ((bitcast<u32>(_e2335_) & 2139095040u) != 2139095040u) {
                    phi_1336_ = select(true, false, (_e2335_ > 0.000000000001f));
                } else {
                    phi_1336_ = true;
                }
                let _e2342_ = phi_1336_;
                let _e2343_ = select(_e2335_, 0.000000000001f, _e2342_);
                let _e2346_ = uniforms.member.member_4_;
                let _e2347_ = bitcast<u32>(_e2346_);
                let _e2348_ = (_e2347_ & 2139095040u);
                if (_e2348_ == 2139095040u) {
                    phi_1353_ = true;
                } else {
                    phi_1353_ = (_e2348_ == 0u);
                }
                let _e2352_ = phi_1353_;
                let _e2354_ = bitcast<f32>(select(_e2347_, 0u, _e2352_));
                let _e2357_ = uniforms.member.member_5_;
                let _e2358_ = bitcast<u32>(_e2357_);
                let _e2359_ = (_e2358_ & 2139095040u);
                if (_e2359_ == 2139095040u) {
                    phi_1371_ = true;
                } else {
                    phi_1371_ = (_e2359_ == 0u);
                }
                let _e2363_ = phi_1371_;
                let _e2365_ = bitcast<f32>(select(_e2358_, 0u, _e2363_));
                let _e2368_ = uniforms.member.member_6_;
                let _e2369_ = bitcast<u32>(_e2368_);
                let _e2370_ = (_e2369_ & 2139095040u);
                if (_e2370_ == 2139095040u) {
                    phi_1389_ = true;
                } else {
                    phi_1389_ = (_e2370_ == 0u);
                }
                let _e2374_ = phi_1389_;
                let _e2376_ = bitcast<f32>(select(_e2369_, 0u, _e2374_));
                let _e2377_ = (bitcast<f32>(select(_e2173_, 0u, _e2178_)) + _e2343_);
                if ((bitcast<u32>(_e2377_) & 2139095040u) != 2139095040u) {
                    let _e2381_ = (_e2377_ <= 0f);
                    if _e2381_ {
                        phi_1488_ = f32();
                    } else {
                        if (_e2377_ < 0.00001f) {
                            let _e2413_ = (select(0.000000000000000000000001f, _e2377_, (_e2377_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e2413_ != _e2413_) {
                                phi_5157_ = true;
                            } else {
                                phi_5157_ = (0.000001f >= _e2413_);
                            }
                            let _e2417_ = phi_5157_;
                            let _e2418_ = select(_e2413_, 0.000001f, _e2417_);
                            let _e2421_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2418_)));
                            let _e2425_ = (_e2421_ + (_e2421_ * (1f - (_e2418_ * _e2421_))));
                            let _e2429_ = (_e2425_ + (_e2425_ * (1f - (_e2418_ * _e2425_))));
                            let _e2435_ = bitcast<u32>(((_e2429_ + (_e2429_ * (1f - (_e2418_ * _e2429_)))) * 18446744000000000000f));
                            let _e2436_ = (_e2435_ & 2139095040u);
                            if (_e2436_ == 2139095040u) {
                                phi_1446_ = true;
                            } else {
                                phi_1446_ = (_e2436_ == 0u);
                            }
                            let _e2440_ = phi_1446_;
                            phi_1487_ = bitcast<f32>(select(_e2435_, 0u, _e2440_));
                        } else {
                            if (_e2377_ != _e2377_) {
                                phi_5097_ = true;
                            } else {
                                phi_5097_ = (0.000001f >= _e2377_);
                            }
                            let _e2386_ = phi_5097_;
                            let _e2387_ = select(_e2377_, 0.000001f, _e2386_);
                            let _e2390_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2387_)));
                            let _e2394_ = (_e2390_ + (_e2390_ * (1f - (_e2387_ * _e2390_))));
                            let _e2398_ = (_e2394_ + (_e2394_ * (1f - (_e2387_ * _e2394_))));
                            let _e2403_ = bitcast<u32>((_e2398_ + (_e2398_ * (1f - (_e2387_ * _e2398_)))));
                            let _e2404_ = (_e2403_ & 2139095040u);
                            if (_e2404_ == 2139095040u) {
                                phi_1481_ = true;
                            } else {
                                phi_1481_ = (_e2404_ == 0u);
                            }
                            let _e2408_ = phi_1481_;
                            phi_1487_ = bitcast<f32>(select(_e2403_, 0u, _e2408_));
                        }
                        let _e2444_ = phi_1487_;
                        phi_1488_ = _e2444_;
                    }
                    let _e2446_ = phi_1488_;
                    phi_1489_ = _e2446_;
                    phi_1490_ = _e2381_;
                } else {
                    phi_1489_ = f32();
                    phi_1490_ = true;
                }
                let _e2448_ = phi_1489_;
                let _e2450_ = phi_1490_;
                let _e2451_ = select(_e2448_, 0f, _e2450_);
                let _e2452_ = (bitcast<f32>(select(_e2249_, 0u, _e2254_)) + _e2343_);
                if ((bitcast<u32>(_e2452_) & 2139095040u) != 2139095040u) {
                    let _e2456_ = (_e2452_ <= 0f);
                    if _e2456_ {
                        phi_1588_ = f32();
                    } else {
                        if (_e2452_ < 0.00001f) {
                            let _e2488_ = (select(0.000000000000000000000001f, _e2452_, (_e2452_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e2488_ != _e2488_) {
                                phi_5277_ = true;
                            } else {
                                phi_5277_ = (0.000001f >= _e2488_);
                            }
                            let _e2492_ = phi_5277_;
                            let _e2493_ = select(_e2488_, 0.000001f, _e2492_);
                            let _e2496_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2493_)));
                            let _e2500_ = (_e2496_ + (_e2496_ * (1f - (_e2493_ * _e2496_))));
                            let _e2504_ = (_e2500_ + (_e2500_ * (1f - (_e2493_ * _e2500_))));
                            let _e2510_ = bitcast<u32>(((_e2504_ + (_e2504_ * (1f - (_e2493_ * _e2504_)))) * 18446744000000000000f));
                            let _e2511_ = (_e2510_ & 2139095040u);
                            if (_e2511_ == 2139095040u) {
                                phi_1546_ = true;
                            } else {
                                phi_1546_ = (_e2511_ == 0u);
                            }
                            let _e2515_ = phi_1546_;
                            phi_1587_ = bitcast<f32>(select(_e2510_, 0u, _e2515_));
                        } else {
                            if (_e2452_ != _e2452_) {
                                phi_5217_ = true;
                            } else {
                                phi_5217_ = (0.000001f >= _e2452_);
                            }
                            let _e2461_ = phi_5217_;
                            let _e2462_ = select(_e2452_, 0.000001f, _e2461_);
                            let _e2465_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2462_)));
                            let _e2469_ = (_e2465_ + (_e2465_ * (1f - (_e2462_ * _e2465_))));
                            let _e2473_ = (_e2469_ + (_e2469_ * (1f - (_e2462_ * _e2469_))));
                            let _e2478_ = bitcast<u32>((_e2473_ + (_e2473_ * (1f - (_e2462_ * _e2473_)))));
                            let _e2479_ = (_e2478_ & 2139095040u);
                            if (_e2479_ == 2139095040u) {
                                phi_1581_ = true;
                            } else {
                                phi_1581_ = (_e2479_ == 0u);
                            }
                            let _e2483_ = phi_1581_;
                            phi_1587_ = bitcast<f32>(select(_e2478_, 0u, _e2483_));
                        }
                        let _e2519_ = phi_1587_;
                        phi_1588_ = _e2519_;
                    }
                    let _e2521_ = phi_1588_;
                    phi_1589_ = _e2521_;
                    phi_1590_ = _e2456_;
                } else {
                    phi_1589_ = f32();
                    phi_1590_ = true;
                }
                let _e2523_ = phi_1589_;
                let _e2525_ = phi_1590_;
                let _e2526_ = select(_e2523_, 0f, _e2525_);
                let _e2527_ = (bitcast<f32>(select(_e2325_, 0u, _e2330_)) + _e2343_);
                if ((bitcast<u32>(_e2527_) & 2139095040u) != 2139095040u) {
                    let _e2531_ = (_e2527_ <= 0f);
                    if _e2531_ {
                        phi_1688_ = f32();
                    } else {
                        if (_e2527_ < 0.00001f) {
                            let _e2563_ = (select(0.000000000000000000000001f, _e2527_, (_e2527_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e2563_ != _e2563_) {
                                phi_5397_ = true;
                            } else {
                                phi_5397_ = (0.000001f >= _e2563_);
                            }
                            let _e2567_ = phi_5397_;
                            let _e2568_ = select(_e2563_, 0.000001f, _e2567_);
                            let _e2571_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2568_)));
                            let _e2575_ = (_e2571_ + (_e2571_ * (1f - (_e2568_ * _e2571_))));
                            let _e2579_ = (_e2575_ + (_e2575_ * (1f - (_e2568_ * _e2575_))));
                            let _e2585_ = bitcast<u32>(((_e2579_ + (_e2579_ * (1f - (_e2568_ * _e2579_)))) * 18446744000000000000f));
                            let _e2586_ = (_e2585_ & 2139095040u);
                            if (_e2586_ == 2139095040u) {
                                phi_1646_ = true;
                            } else {
                                phi_1646_ = (_e2586_ == 0u);
                            }
                            let _e2590_ = phi_1646_;
                            phi_1687_ = bitcast<f32>(select(_e2585_, 0u, _e2590_));
                        } else {
                            if (_e2527_ != _e2527_) {
                                phi_5337_ = true;
                            } else {
                                phi_5337_ = (0.000001f >= _e2527_);
                            }
                            let _e2536_ = phi_5337_;
                            let _e2537_ = select(_e2527_, 0.000001f, _e2536_);
                            let _e2540_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2537_)));
                            let _e2544_ = (_e2540_ + (_e2540_ * (1f - (_e2537_ * _e2540_))));
                            let _e2548_ = (_e2544_ + (_e2544_ * (1f - (_e2537_ * _e2544_))));
                            let _e2553_ = bitcast<u32>((_e2548_ + (_e2548_ * (1f - (_e2537_ * _e2548_)))));
                            let _e2554_ = (_e2553_ & 2139095040u);
                            if (_e2554_ == 2139095040u) {
                                phi_1681_ = true;
                            } else {
                                phi_1681_ = (_e2554_ == 0u);
                            }
                            let _e2558_ = phi_1681_;
                            phi_1687_ = bitcast<f32>(select(_e2553_, 0u, _e2558_));
                        }
                        let _e2594_ = phi_1687_;
                        phi_1688_ = _e2594_;
                    }
                    let _e2596_ = phi_1688_;
                    phi_1689_ = _e2596_;
                    phi_1690_ = _e2531_;
                } else {
                    phi_1689_ = f32();
                    phi_1690_ = true;
                }
                let _e2598_ = phi_1689_;
                let _e2600_ = phi_1690_;
                let _e2601_ = select(_e2598_, 0f, _e2600_);
                let _e2603_ = bitcast<u32>((bitcast<f32>(select(_e2135_, 0u, _e2140_)) * _e2451_));
                let _e2604_ = (_e2603_ & 2139095040u);
                if (_e2604_ == 2139095040u) {
                    phi_1706_ = true;
                } else {
                    phi_1706_ = (_e2604_ == 0u);
                }
                let _e2608_ = phi_1706_;
                let _e2610_ = bitcast<f32>(select(_e2603_, 0u, _e2608_));
                let _e2612_ = bitcast<u32>((bitcast<f32>(select(_e2211_, 0u, _e2216_)) * _e2526_));
                let _e2613_ = (_e2612_ & 2139095040u);
                if (_e2613_ == 2139095040u) {
                    phi_1723_ = true;
                } else {
                    phi_1723_ = (_e2613_ == 0u);
                }
                let _e2617_ = phi_1723_;
                let _e2619_ = bitcast<f32>(select(_e2612_, 0u, _e2617_));
                let _e2621_ = bitcast<u32>((bitcast<f32>(select(_e2287_, 0u, _e2292_)) * _e2601_));
                let _e2622_ = (_e2621_ & 2139095040u);
                if (_e2622_ == 2139095040u) {
                    phi_1740_ = true;
                } else {
                    phi_1740_ = (_e2622_ == 0u);
                }
                let _e2626_ = phi_1740_;
                let _e2628_ = bitcast<f32>(select(_e2621_, 0u, _e2626_));
                let _e2634_ = bitcast<u32>((((_e2354_ * _e2610_) + (_e2365_ * _e2619_)) + (_e2376_ * _e2628_)));
                let _e2635_ = (_e2634_ & 2139095040u);
                if (_e2635_ == 2139095040u) {
                    phi_1761_ = true;
                } else {
                    phi_1761_ = (_e2635_ == 0u);
                }
                let _e2639_ = phi_1761_;
                if (_e82_ == 0u) {
                    if (_e81_.x < _e1713_) {
                        if (_e81_.y < _e1706_) {
                            let _e2647_ = ((_e81_.y * _e1713_) + _e81_.x);
                            if (_e93_ == 0u) {
                                phi_1792_ = 0u;
                            } else {
                                let _e2649_ = (_e93_ - 1u);
                                phi_1792_ = select(_e2649_, _e2647_, (_e2647_ < _e2649_));
                            }
                            let _e2653_ = phi_1792_;
                            if (_e2653_ < _e93_) {
                            } else {
                                break;
                            }
                            band_losses.member[_e2653_] = vec4<f32>(_e2610_, _e2619_, _e2628_, bitcast<f32>(select(_e2634_, 0u, _e2639_)));
                        }
                    }
                }
                let _e2658_ = ((2f * _e2354_) * _e2451_);
                let _e2660_ = ((2f * _e2365_) * _e2526_);
                let _e2662_ = ((2f * _e2376_) * _e2601_);
                if _e2018_ {
                } else {
                    break;
                }
                let _e2665_ = buf_b[_e2017_].x;
                let _e2668_ = buf_b[_e2017_].y;
                let _e2671_ = buf_b[_e2017_].z;
                let _e2674_ = buf_b[_e2017_].w;
                if _e2018_ {
                } else {
                    break;
                }
                let _e2675_ = buf_c[_e2017_].x;
                let _e2676_ = buf_c[_e2017_].y;
                if _e2018_ {
                } else {
                    break;
                }
                buf_b[_e2017_] = vec4<f32>((_e2665_ * _e2658_), (_e2668_ * _e2658_), (_e2671_ * _e2660_), (_e2674_ * _e2660_));
                if _e2018_ {
                } else {
                    break;
                }
                buf_c[_e2017_].x = (_e2675_ * _e2662_);
                buf_c[_e2017_].y = (_e2676_ * _e2662_);
                workgroupBarrier();
                if _e1731_ {
                    phi_1874_ = type_22_(0u, _e1727_);
                    phi_1877_ = 0f;
                    phi_1879_ = 0f;
                    loop {
                        let _e2686_ = phi_1874_;
                        let _e2688_ = phi_1877_;
                        let _e2690_ = phi_1879_;
                        local_15_ = _e2690_;
                        local_16_ = _e2688_;
                        if (_e2686_.member < _e2686_.member_1_) {
                            phi_1875_ = type_22_((_e2686_.member + 1u), _e2686_.member_1_);
                            phi_1902_ = type_22_(1u, _e2686_.member);
                        } else {
                            phi_1875_ = _e2686_;
                            phi_1902_ = type_22_(0u, u32());
                        }
                        let _e2703_ = phi_1875_;
                        let _e2705_ = phi_1902_;
                        let _e2709_ = (bitcast<i32>(_e2705_.member) != 0i);
                        if _e2709_ {
                            let _e2710_ = (_e82_ + _e2705_.member_1_);
                            if (_e2710_ < _e1720_) {
                                if (_e89_ == 0u) {
                                    phi_1925_ = 0u;
                                } else {
                                    let _e2713_ = (_e89_ - 1u);
                                    phi_1925_ = select(_e2713_, _e2705_.member_1_, (_e2705_.member_1_ < _e2713_));
                                }
                                let _e2717_ = phi_1925_;
                                if (_e2717_ < _e89_) {
                                } else {
                                    phi_5825_ = true;
                                    break;
                                }
                                let _e2722_ = bezout_taps.member[_e2717_].x;
                                let _e2726_ = bezout_taps.member[_e2717_].y;
                                let _e2730_ = bezout_taps.member[_e2717_].z;
                                let _e2732_ = select(255u, _e2710_, (_e2710_ < 255u));
                                let _e2733_ = (_e2732_ < 256u);
                                if _e2733_ {
                                } else {
                                    phi_5825_ = true;
                                    break;
                                }
                                let _e2736_ = buf_b[_e2732_].x;
                                let _e2739_ = buf_b[_e2732_].y;
                                let _e2742_ = buf_b[_e2732_].z;
                                let _e2745_ = buf_b[_e2732_].w;
                                if _e2733_ {
                                } else {
                                    phi_5825_ = true;
                                    break;
                                }
                                let _e2748_ = buf_c[_e2732_].x;
                                let _e2751_ = buf_c[_e2732_].y;
                                let _e2752_ = bitcast<u32>(_e2722_);
                                let _e2753_ = (_e2752_ & 2139095040u);
                                let _e2754_ = (_e2753_ == 2139095040u);
                                if _e2754_ {
                                    phi_2010_ = true;
                                } else {
                                    phi_2010_ = (_e2753_ == 0u);
                                }
                                let _e2757_ = phi_2010_;
                                let _e2761_ = bitcast<u32>(_e2726_);
                                let _e2762_ = (_e2761_ & 2139095040u);
                                let _e2763_ = (_e2762_ == 2139095040u);
                                if _e2763_ {
                                    phi_2027_ = true;
                                } else {
                                    phi_2027_ = (_e2762_ == 0u);
                                }
                                let _e2766_ = phi_2027_;
                                let _e2771_ = bitcast<u32>(_e2730_);
                                let _e2772_ = (_e2771_ & 2139095040u);
                                let _e2773_ = (_e2772_ == 2139095040u);
                                if _e2773_ {
                                    phi_2045_ = true;
                                } else {
                                    phi_2045_ = (_e2772_ == 0u);
                                }
                                let _e2776_ = phi_2045_;
                                if _e2754_ {
                                    phi_2064_ = true;
                                } else {
                                    phi_2064_ = (_e2753_ == 0u);
                                }
                                let _e2784_ = phi_2064_;
                                if _e2763_ {
                                    phi_2081_ = true;
                                } else {
                                    phi_2081_ = (_e2762_ == 0u);
                                }
                                let _e2790_ = phi_2081_;
                                if _e2773_ {
                                    phi_2099_ = true;
                                } else {
                                    phi_2099_ = (_e2772_ == 0u);
                                }
                                let _e2797_ = phi_2099_;
                                phi_2108_ = (_e2688_ + (((bitcast<f32>(select(_e2752_, 0u, _e2784_)) * _e2739_) + (bitcast<f32>(select(_e2761_, 0u, _e2790_)) * _e2745_)) + (bitcast<f32>(select(_e2771_, 0u, _e2797_)) * _e2751_)));
                                phi_2109_ = (_e2690_ + (((bitcast<f32>(select(_e2752_, 0u, _e2757_)) * _e2736_) + (bitcast<f32>(select(_e2761_, 0u, _e2766_)) * _e2742_)) + (bitcast<f32>(select(_e2771_, 0u, _e2776_)) * _e2748_)));
                            } else {
                                phi_2108_ = _e2688_;
                                phi_2109_ = _e2690_;
                            }
                            let _e2804_ = phi_2108_;
                            let _e2806_ = phi_2109_;
                            phi_1878_ = _e2804_;
                            phi_1880_ = _e2806_;
                        } else {
                            phi_1878_ = f32();
                            phi_1880_ = f32();
                        }
                        let _e2808_ = phi_1878_;
                        let _e2810_ = phi_1880_;
                        phi_1874_ = _e2703_;
                        phi_1877_ = _e2808_;
                        phi_1879_ = _e2810_;
                        phi_5825_ = _e2323_;
                        if !(_e2709_) {
                            break;
                        }
                    }
                    let _e2813_ = phi_5825_;
                    if _e2813_ {
                        break;
                    }
                    let _e2815_ = local_15_;
                    let _e2816_ = bitcast<u32>(_e2815_);
                    let _e2817_ = (_e2816_ & 2139095040u);
                    if (_e2817_ == 2139095040u) {
                        phi_2120_ = true;
                    } else {
                        phi_2120_ = (_e2817_ == 0u);
                    }
                    let _e2821_ = phi_2120_;
                    let _e2825_ = local_16_;
                    let _e2826_ = bitcast<u32>(_e2825_);
                    let _e2827_ = (_e2826_ & 2139095040u);
                    if (_e2827_ == 2139095040u) {
                        phi_2136_ = true;
                    } else {
                        phi_2136_ = (_e2827_ == 0u);
                    }
                    let _e2831_ = phi_2136_;
                    if (_e91_ == 0u) {
                        phi_2152_ = 0u;
                    } else {
                        let _e2835_ = (_e91_ - 1u);
                        phi_2152_ = select(_e2835_, _e1734_, (_e1734_ < _e2835_));
                    }
                    let _e2839_ = phi_2152_;
                    if (_e2839_ < _e91_) {
                    } else {
                        break;
                    }
                    d_loss_d_pred.member[_e2839_].x = bitcast<f32>(select(_e2816_, 0u, _e2821_));
                    d_loss_d_pred.member[_e2839_].y = bitcast<f32>(select(_e2826_, 0u, _e2831_));
                }
            } else {
                phi_2164_ = 4u;
                phi_2167_ = 16u;
                loop {
                    let _e101_ = phi_2164_;
                    let _e103_ = phi_2167_;
                    local = _e103_;
                    local_1_ = _e103_;
                    local_2_ = _e103_;
                    local_3_ = _e103_;
                    local_4_ = _e101_;
                    local_5_ = _e101_;
                    local_8_ = _e101_;
                    local_9_ = _e101_;
                    local_12_ = _e101_;
                    if (_e103_ < _e98_) {
                        let _e105_ = (_e103_ < 256u);
                        if _e105_ {
                            phi_2181_ = (_e101_ + 1u);
                            phi_2182_ = (_e103_ << bitcast<u32>(1i));
                        } else {
                            phi_2181_ = u32();
                            phi_2182_ = u32();
                        }
                        let _e110_ = phi_2181_;
                        let _e112_ = phi_2182_;
                        phi_2165_ = _e110_;
                        phi_2168_ = _e112_;
                        phi_2183_ = _e105_;
                    } else {
                        phi_2165_ = u32();
                        phi_2168_ = u32();
                        phi_2183_ = false;
                    }
                    let _e114_ = phi_2165_;
                    let _e116_ = phi_2168_;
                    let _e118_ = phi_2183_;
                    phi_2164_ = _e114_;
                    phi_2167_ = _e116_;
                    if !(_e118_) {
                        break;
                    }
                }
                let _e122_ = uniforms.member.member;
                let _e125_ = uniforms.member.member_3_;
                let _e127_ = select(1u, _e125_, (1u < _e125_));
                if (_e127_ <= 1u) {
                    phi_2252_ = select(_e93_, 0u, (_e127_ == 0u));
                } else {
                    if (_e93_ < _e127_) {
                        phi_2251_ = 0u;
                    } else {
                        phi_2211_ = 1u;
                        phi_2214_ = _e127_;
                        loop {
                            let _e131_ = phi_2211_;
                            let _e133_ = phi_2214_;
                            local_17_ = _e131_;
                            local_18_ = _e133_;
                            let _e136_ = (_e133_ <= (_e93_ >> bitcast<u32>(1i)));
                            if _e136_ {
                                phi_2212_ = (_e131_ << bitcast<u32>(1i));
                                phi_2215_ = (_e133_ << bitcast<u32>(1i));
                            } else {
                                phi_2212_ = u32();
                                phi_2215_ = u32();
                            }
                            let _e142_ = phi_2212_;
                            let _e144_ = phi_2215_;
                            phi_2211_ = _e142_;
                            phi_2214_ = _e144_;
                            if !(_e136_) {
                                break;
                            }
                        }
                        let _e2860_ = local_17_;
                        phi_2226_ = _e2860_;
                        phi_2229_ = 0u;
                        let _e2864_ = local_18_;
                        phi_2231_ = _e2864_;
                        phi_2233_ = _e93_;
                        loop {
                            let _e147_ = phi_2226_;
                            let _e149_ = phi_2229_;
                            let _e151_ = phi_2231_;
                            let _e153_ = phi_2233_;
                            local_19_ = _e149_;
                            let _e154_ = (_e147_ > 0u);
                            if _e154_ {
                                if (_e153_ >= _e151_) {
                                    phi_2247_ = (_e149_ | _e147_);
                                    phi_2248_ = (_e153_ - _e151_);
                                } else {
                                    phi_2247_ = _e149_;
                                    phi_2248_ = _e153_;
                                }
                                let _e159_ = phi_2247_;
                                let _e161_ = phi_2248_;
                                phi_2227_ = (_e147_ >> bitcast<u32>(1i));
                                phi_2230_ = _e159_;
                                phi_2232_ = (_e151_ >> bitcast<u32>(1i));
                                phi_2234_ = _e161_;
                            } else {
                                phi_2227_ = u32();
                                phi_2230_ = u32();
                                phi_2232_ = u32();
                                phi_2234_ = u32();
                            }
                            let _e167_ = phi_2227_;
                            let _e169_ = phi_2230_;
                            let _e171_ = phi_2232_;
                            let _e173_ = phi_2234_;
                            phi_2226_ = _e167_;
                            phi_2229_ = _e169_;
                            phi_2231_ = _e171_;
                            phi_2233_ = _e173_;
                            if !(_e154_) {
                                break;
                            }
                        }
                        let _e2874_ = local_19_;
                        phi_2251_ = _e2874_;
                    }
                    let _e176_ = phi_2251_;
                    phi_2252_ = _e176_;
                }
                let _e180_ = phi_2252_;
                let _e182_ = select(1u, _e180_, (1u < _e180_));
                let _e185_ = uniforms.member.member_2_;
                let _e187_ = local;
                let _e190_ = local_1_;
                let _e194_ = local_2_;
                let _e196_ = (_e194_ >> bitcast<u32>(1i));
                if (_e89_ == 0u) {
                    phi_2277_ = 0u;
                } else {
                    let _e198_ = (_e89_ - 1u);
                    phi_2277_ = select(_e198_, _e82_, (_e82_ < _e198_));
                }
                let _e202_ = phi_2277_;
                if (_e202_ < _e89_) {
                } else {
                    break;
                }
                let _e207_ = bezout_taps.member[_e202_].x;
                let _e211_ = bezout_taps.member[_e202_].y;
                let _e215_ = bezout_taps.member[_e202_].z;
                let _e219_ = bezout_taps.member[_e202_].w;
                let _e221_ = local_3_;
                let _e222_ = (_e82_ < _e221_);
                if _e222_ {
                    let _e223_ = bitcast<u32>(_e207_);
                    let _e224_ = (_e223_ & 2139095040u);
                    if (_e224_ == 2139095040u) {
                        phi_2312_ = true;
                    } else {
                        phi_2312_ = (_e224_ == 0u);
                    }
                    let _e228_ = phi_2312_;
                    phi_2318_ = bitcast<f32>(select(_e223_, 0u, _e228_));
                } else {
                    phi_2318_ = 0f;
                }
                let _e232_ = phi_2318_;
                let _e234_ = select(255u, _e82_, (_e82_ < 255u));
                let _e235_ = (_e234_ < 256u);
                if _e235_ {
                } else {
                    break;
                }
                buf_c[_e234_].x = _e232_;
                buf_c[_e234_].y = 0f;
                let _e240_ = ((_e81_.x * select(_e190_, _e185_, (_e187_ < _e185_))) + _e82_);
                if _e222_ {
                    let _e241_ = (_e240_ < _e122_);
                    if _e241_ {
                        let _e242_ = (_e81_.y < _e127_);
                        let _e244_ = ((_e81_.y * _e122_) + _e240_);
                        if _e242_ {
                            if (_e85_ == 0u) {
                                phi_2363_ = 0u;
                            } else {
                                let _e246_ = (_e85_ - 1u);
                                phi_2363_ = select(_e246_, _e244_, (_e244_ < _e246_));
                            }
                            let _e250_ = phi_2363_;
                            if (_e250_ < _e85_) {
                            } else {
                                break;
                            }
                            let _e255_ = pred_audio.member[_e250_].x;
                            let _e259_ = pred_audio.member[_e250_].y;
                            if (_e87_ == 0u) {
                                phi_2386_ = 0u;
                            } else {
                                let _e261_ = (_e87_ - 1u);
                                phi_2386_ = select(_e261_, _e244_, (_e244_ < _e261_));
                            }
                            let _e265_ = phi_2386_;
                            if (_e265_ < _e87_) {
                            } else {
                                break;
                            }
                            let _e270_ = target_audio.member[_e265_].x;
                            let _e274_ = target_audio.member[_e265_].y;
                            let _e277_ = bitcast<u32>(((_e255_ - _e270_) * _e232_));
                            let _e278_ = (_e277_ & 2139095040u);
                            if (_e278_ == 2139095040u) {
                                phi_2411_ = true;
                            } else {
                                phi_2411_ = (_e278_ == 0u);
                            }
                            let _e282_ = phi_2411_;
                            let _e287_ = bitcast<u32>(((_e259_ - _e274_) * _e232_));
                            let _e288_ = (_e287_ & 2139095040u);
                            if (_e288_ == 2139095040u) {
                                phi_2429_ = true;
                            } else {
                                phi_2429_ = (_e288_ == 0u);
                            }
                            let _e292_ = phi_2429_;
                            let _e296_ = bitcast<u32>((_e270_ * _e232_));
                            let _e297_ = (_e296_ & 2139095040u);
                            if (_e297_ == 2139095040u) {
                                phi_2446_ = true;
                            } else {
                                phi_2446_ = (_e297_ == 0u);
                            }
                            let _e301_ = phi_2446_;
                            let _e305_ = bitcast<u32>((_e274_ * _e232_));
                            let _e306_ = (_e305_ & 2139095040u);
                            if (_e306_ == 2139095040u) {
                                phi_2463_ = true;
                            } else {
                                phi_2463_ = (_e306_ == 0u);
                            }
                            let _e310_ = phi_2463_;
                            phi_2469_ = bitcast<f32>(select(_e305_, 0u, _e310_));
                            phi_2470_ = bitcast<f32>(select(_e296_, 0u, _e301_));
                            phi_2471_ = bitcast<f32>(select(_e287_, 0u, _e292_));
                            phi_2472_ = bitcast<f32>(select(_e277_, 0u, _e282_));
                        } else {
                            phi_2469_ = f32();
                            phi_2470_ = f32();
                            phi_2471_ = f32();
                            phi_2472_ = f32();
                        }
                        let _e314_ = phi_2469_;
                        let _e316_ = phi_2470_;
                        let _e318_ = phi_2471_;
                        let _e320_ = phi_2472_;
                        phi_2474_ = _e314_;
                        phi_2475_ = _e316_;
                        phi_2476_ = _e318_;
                        phi_2477_ = _e320_;
                        phi_2478_ = _e244_;
                        phi_2479_ = _e242_;
                        phi_2483_ = select(true, false, _e242_);
                    } else {
                        phi_2474_ = f32();
                        phi_2475_ = f32();
                        phi_2476_ = f32();
                        phi_2477_ = f32();
                        phi_2478_ = u32();
                        phi_2479_ = bool();
                        phi_2483_ = false;
                    }
                    let _e323_ = phi_2474_;
                    let _e325_ = phi_2475_;
                    let _e327_ = phi_2476_;
                    let _e329_ = phi_2477_;
                    let _e331_ = phi_2478_;
                    let _e333_ = phi_2479_;
                    let _e335_ = phi_2483_;
                    phi_2484_ = _e323_;
                    phi_2485_ = _e325_;
                    phi_2486_ = _e327_;
                    phi_2487_ = _e329_;
                    phi_2488_ = _e331_;
                    phi_2489_ = _e333_;
                    phi_2492_ = select(true, false, _e241_);
                    phi_2493_ = _e335_;
                } else {
                    phi_2484_ = f32();
                    phi_2485_ = f32();
                    phi_2486_ = f32();
                    phi_2487_ = f32();
                    phi_2488_ = u32();
                    phi_2489_ = bool();
                    phi_2492_ = true;
                    phi_2493_ = false;
                }
                let _e338_ = phi_2484_;
                let _e340_ = phi_2485_;
                let _e342_ = phi_2486_;
                let _e344_ = phi_2487_;
                let _e346_ = phi_2488_;
                let _e348_ = phi_2489_;
                let _e350_ = phi_2492_;
                let _e352_ = phi_2493_;
                if _e350_ {
                    phi_2499_ = ((_e81_.y * _e122_) + _e240_);
                } else {
                    phi_2499_ = _e346_;
                }
                let _e356_ = phi_2499_;
                let _e358_ = select(_e352_, true, _e350_);
                let _e363_ = select(_e346_, _e356_, _e358_);
                if _e222_ {
                    phi_2516_ = 0u;
                    phi_2519_ = 0u;
                    phi_2521_ = _e82_;
                    loop {
                        let _e366_ = phi_2516_;
                        let _e368_ = phi_2519_;
                        let _e370_ = phi_2521_;
                        local_20_ = _e368_;
                        let _e372_ = local_4_;
                        let _e373_ = (_e366_ < _e372_);
                        if _e373_ {
                            phi_2517_ = (_e366_ + 1u);
                            phi_2520_ = ((_e368_ << bitcast<u32>(1i)) | (_e370_ & 1u));
                            phi_2522_ = (_e370_ >> bitcast<u32>(1i));
                        } else {
                            phi_2517_ = u32();
                            phi_2520_ = u32();
                            phi_2522_ = u32();
                        }
                        let _e382_ = phi_2517_;
                        let _e384_ = phi_2520_;
                        let _e386_ = phi_2522_;
                        phi_2516_ = _e382_;
                        phi_2519_ = _e384_;
                        phi_2521_ = _e386_;
                        if !(_e373_) {
                            break;
                        }
                    }
                    let _e2918_ = local_20_;
                    phi_2534_ = _e2918_;
                } else {
                    phi_2534_ = _e82_;
                }
                let _e389_ = phi_2534_;
                let _e392_ = select(255u, _e389_, (_e389_ < 255u));
                let _e393_ = (_e392_ < 256u);
                if _e393_ {
                } else {
                    break;
                }
                buf_a[_e392_] = vec4<f32>(select(_e344_, 0f, _e358_), 0f, select(_e342_, 0f, _e358_), 0f);
                if _e393_ {
                } else {
                    break;
                }
                buf_b[_e392_] = vec4<f32>(select(_e340_, 0f, _e358_), 0f, select(_e338_, 0f, _e358_), 0f);
                workgroupBarrier();
                phi_5453_ = false;
                phi_2572_ = 0u;
                loop {
                    let _e398_ = phi_5453_;
                    let _e400_ = phi_2572_;
                    let _e402_ = local_5_;
                    let _e403_ = (_e400_ < _e402_);
                    if _e403_ {
                        let _e404_ = (_e400_ & 31u);
                        let _e406_ = (1u << bitcast<u32>(_e404_));
                        if (_e82_ < _e196_) {
                            let _e412_ = (_e82_ & (_e406_ - 1u));
                            let _e417_ = (((_e82_ >> bitcast<u32>(_e404_)) << bitcast<u32>(((_e400_ + 1u) & 31u))) + _e412_);
                            let _e418_ = (_e417_ + _e406_);
                            let _e422_ = f32((_e412_ << bitcast<u32>(((31u - _e400_) & 31u))));
                            let _e423_ = (_e422_ * 0.0000000014629181f);
                            let _e424_ = bitcast<u32>(_e423_);
                            if ((_e424_ & 2139095040u) != 2139095040u) {
                                if (_e423_ == 0f) {
                                    phi_5514_ = _e398_;
                                    phi_2826_ = type_27_();
                                    phi_2827_ = true;
                                } else {
                                    let _e430_ = (bitcast<f32>((_e424_ & 2147483647u)) > 65536f);
                                    if _e430_ {
                                        phi_5515_ = _e398_;
                                        phi_2825_ = type_27_();
                                    } else {
                                        let _e435_ = (bitcast<i32>(bitcast<u32>(((_e422_ * 0.0000000009313226f) + 12582912f))) - 1262485504i);
                                        let _e436_ = f32(_e435_);
                                        let _e442_ = (((_e423_ - (_e436_ * 1.5703125f)) - (_e436_ * 0.0004837513f)) - (_e436_ * 0.0000000754979f));
                                        let _e443_ = bitcast<u32>(_e442_);
                                        if ((_e443_ & 2139095040u) != 2139095040u) {
                                            let _e448_ = (bitcast<f32>((_e443_ & 2147483647u)) > 0.85f);
                                            if _e448_ {
                                                phi_5518_ = _e398_;
                                                phi_2817_ = type_27_();
                                            } else {
                                                let _e449_ = (_e442_ * _e442_);
                                                let _e462_ = (_e442_ + ((_e442_ * _e449_) * (-0.16666667f + (_e449_ * (0.008333334f + (_e449_ * (-0.0001984127f + (_e449_ * (0.0000027557319f + (_e449_ * (-0.000000025052108f + (_e449_ * 0.00000000016059044f))))))))))));
                                                let _e475_ = (_e449_ * (-0.5f + (_e449_ * (0.041666668f + (_e449_ * (-0.0013888889f + (_e449_ * (0.000024801588f + (_e449_ * (-0.0000002755732f + (_e449_ * (0.0000000020876756f + (_e449_ * -0.000000000011470746f)))))))))))));
                                                let _e476_ = (1f + _e475_);
                                                switch (_e435_ & 3i) {
                                                    case 0: {
                                                        phi_2681_ = _e462_;
                                                        phi_2682_ = _e476_;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_2681_ = _e476_;
                                                        phi_2682_ = -(_e462_);
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_2681_ = -(_e462_);
                                                        phi_2682_ = (-1f - _e475_);
                                                        break;
                                                    }
                                                    default: {
                                                        phi_2681_ = (-1f - _e475_);
                                                        phi_2682_ = _e462_;
                                                        break;
                                                    }
                                                }
                                                let _e483_ = phi_2681_;
                                                let _e485_ = phi_2682_;
                                                let _e486_ = bitcast<u32>(_e485_);
                                                let _e487_ = bitcast<u32>(_e483_);
                                                if ((_e486_ & 2147483647u) == 0u) {
                                                    phi_2692_ = true;
                                                } else {
                                                    phi_2692_ = ((_e487_ & 2147483647u) == 0u);
                                                }
                                                let _e493_ = phi_2692_;
                                                phi_2695_ = type_32_(0i, 2i);
                                                phi_2698_ = _e483_;
                                                phi_2700_ = _e485_;
                                                phi_2702_ = _e487_;
                                                phi_2704_ = _e486_;
                                                loop {
                                                    let _e495_ = phi_2695_;
                                                    let _e497_ = phi_2698_;
                                                    let _e499_ = phi_2700_;
                                                    let _e501_ = phi_2702_;
                                                    let _e503_ = phi_2704_;
                                                    local_6_ = _e499_;
                                                    local_7_ = _e497_;
                                                    if (_e495_.member < _e495_.member_1_) {
                                                        let _e510_ = (_e495_.member + 1i);
                                                        if select(false, true, ((false == (_e510_ > _e495_.member)) != false)) {
                                                            phi_2731_ = type_33_(0u, i32());
                                                        } else {
                                                            phi_2731_ = type_33_(1u, _e510_);
                                                        }
                                                        let _e520_ = phi_2731_;
                                                        if (bitcast<i32>(_e520_.member) != 0i) {
                                                        } else {
                                                            phi_5447_ = true;
                                                            break;
                                                        }
                                                        phi_2696_ = type_32_(_e520_.member_1_, _e495_.member_1_);
                                                        phi_2745_ = type_33_(1u, _e495_.member);
                                                    } else {
                                                        phi_2696_ = _e495_;
                                                        phi_2745_ = type_33_(0u, i32());
                                                    }
                                                    let _e530_ = phi_2696_;
                                                    let _e532_ = phi_2745_;
                                                    if (bitcast<i32>(_e532_.member) != 0i) {
                                                        let _e538_ = ((_e499_ * _e499_) + (_e497_ * _e497_));
                                                        if (_e538_ > 1f) {
                                                            phi_2769_ = true;
                                                        } else {
                                                            if _e493_ {
                                                                phi_2767_ = false;
                                                            } else {
                                                                phi_2767_ = (_e538_ == 1f);
                                                            }
                                                            let _e542_ = phi_2767_;
                                                            phi_2769_ = _e542_;
                                                        }
                                                        let _e544_ = phi_2769_;
                                                        if _e544_ {
                                                            if ((_e503_ & 2147483647u) >= (_e501_ & 2147483647u)) {
                                                                let _e550_ = (_e503_ - 1u);
                                                                phi_2784_ = _e497_;
                                                                phi_2785_ = bitcast<f32>(_e550_);
                                                                phi_2786_ = _e501_;
                                                                phi_2787_ = _e550_;
                                                            } else {
                                                                let _e548_ = (_e501_ - 1u);
                                                                phi_2784_ = bitcast<f32>(_e548_);
                                                                phi_2785_ = _e499_;
                                                                phi_2786_ = _e548_;
                                                                phi_2787_ = _e503_;
                                                            }
                                                            let _e553_ = phi_2784_;
                                                            let _e555_ = phi_2785_;
                                                            let _e557_ = phi_2786_;
                                                            let _e559_ = phi_2787_;
                                                            phi_2788_ = _e553_;
                                                            phi_2789_ = _e555_;
                                                            phi_2790_ = _e557_;
                                                            phi_2791_ = _e559_;
                                                        } else {
                                                            phi_2788_ = f32();
                                                            phi_2789_ = f32();
                                                            phi_2790_ = u32();
                                                            phi_2791_ = u32();
                                                        }
                                                        let _e561_ = phi_2788_;
                                                        let _e563_ = phi_2789_;
                                                        let _e565_ = phi_2790_;
                                                        let _e567_ = phi_2791_;
                                                        phi_2699_ = _e561_;
                                                        phi_2701_ = _e563_;
                                                        phi_2703_ = _e565_;
                                                        phi_2705_ = _e567_;
                                                        phi_2793_ = _e544_;
                                                    } else {
                                                        phi_2699_ = f32();
                                                        phi_2701_ = f32();
                                                        phi_2703_ = u32();
                                                        phi_2705_ = u32();
                                                        phi_2793_ = false;
                                                    }
                                                    let _e569_ = phi_2699_;
                                                    let _e571_ = phi_2701_;
                                                    let _e573_ = phi_2703_;
                                                    let _e575_ = phi_2705_;
                                                    let _e577_ = phi_2793_;
                                                    phi_2695_ = _e530_;
                                                    phi_2698_ = _e569_;
                                                    phi_2700_ = _e571_;
                                                    phi_2702_ = _e573_;
                                                    phi_2704_ = _e575_;
                                                    phi_5447_ = _e398_;
                                                    if !(_e577_) {
                                                        break;
                                                    }
                                                }
                                                let _e580_ = phi_5447_;
                                                phi_5505_ = _e580_;
                                                if _e580_ {
                                                    break;
                                                }
                                                let _e582_ = local_6_;
                                                let _e583_ = bitcast<u32>(_e582_);
                                                let _e590_ = local_7_;
                                                let _e591_ = bitcast<u32>(_e590_);
                                                phi_5518_ = _e580_;
                                                phi_2817_ = type_27_(bitcast<f32>((_e583_ & select(4294967295u, 2147483648u, ((_e583_ & 2139095040u) == 0u)))), bitcast<f32>((_e591_ & select(4294967295u, 2147483648u, ((_e591_ & 2139095040u) == 0u)))));
                                            }
                                            let _e599_ = phi_5518_;
                                            let _e601_ = phi_2817_;
                                            phi_5517_ = _e599_;
                                            phi_2818_ = _e601_;
                                            phi_2819_ = _e448_;
                                        } else {
                                            phi_5517_ = _e398_;
                                            phi_2818_ = type_27_();
                                            phi_2819_ = true;
                                        }
                                        let _e603_ = phi_5517_;
                                        let _e605_ = phi_2818_;
                                        let _e607_ = phi_2819_;
                                        if _e607_ {
                                            phi_2824_ = type_27_(1f, 0f);
                                        } else {
                                            phi_2824_ = _e605_;
                                        }
                                        let _e609_ = phi_2824_;
                                        phi_5515_ = _e603_;
                                        phi_2825_ = _e609_;
                                    }
                                    let _e611_ = phi_5515_;
                                    let _e613_ = phi_2825_;
                                    phi_5514_ = _e611_;
                                    phi_2826_ = _e613_;
                                    phi_2827_ = _e430_;
                                }
                                let _e615_ = phi_5514_;
                                let _e617_ = phi_2826_;
                                let _e619_ = phi_2827_;
                                phi_5513_ = _e615_;
                                phi_2828_ = _e617_;
                                phi_2829_ = _e619_;
                            } else {
                                phi_5513_ = _e398_;
                                phi_2828_ = type_27_();
                                phi_2829_ = true;
                            }
                            let _e621_ = phi_5513_;
                            let _e623_ = phi_2828_;
                            let _e625_ = phi_2829_;
                            if _e625_ {
                                phi_2834_ = type_27_(1f, 0f);
                            } else {
                                phi_2834_ = _e623_;
                            }
                            let _e627_ = phi_2834_;
                            let _e630_ = -(_e627_.member_1_);
                            let _e632_ = select(255u, _e417_, (_e417_ < 255u));
                            let _e634_ = select(255u, _e418_, (_e418_ < 255u));
                            let _e635_ = (_e632_ < 256u);
                            if _e635_ {
                            } else {
                                phi_5505_ = true;
                                break;
                            }
                            let _e637_ = buf_a[_e632_];
                            let _e638_ = (_e634_ < 256u);
                            if _e638_ {
                            } else {
                                phi_5505_ = true;
                                break;
                            }
                            let _e640_ = buf_a[_e634_];
                            let _e657_ = vec4<f32>(((_e627_.member * _e640_.x) - (_e630_ * _e640_.y)), ((_e627_.member * _e640_.y) + (_e630_ * _e640_.x)), ((_e627_.member * _e640_.z) - (_e630_ * _e640_.w)), ((_e627_.member * _e640_.w) + (_e630_ * _e640_.z)));
                            if _e635_ {
                            } else {
                                phi_5505_ = true;
                                break;
                            }
                            buf_a[_e632_] = (_e637_ + _e657_);
                            if _e638_ {
                            } else {
                                phi_5505_ = true;
                                break;
                            }
                            buf_a[_e634_] = (_e637_ - _e657_);
                            if _e635_ {
                            } else {
                                phi_5505_ = true;
                                break;
                            }
                            let _e661_ = buf_b[_e632_];
                            if _e638_ {
                            } else {
                                phi_5505_ = true;
                                break;
                            }
                            let _e663_ = buf_b[_e634_];
                            let _e680_ = vec4<f32>(((_e627_.member * _e663_.x) - (_e630_ * _e663_.y)), ((_e627_.member * _e663_.y) + (_e630_ * _e663_.x)), ((_e627_.member * _e663_.z) - (_e630_ * _e663_.w)), ((_e627_.member * _e663_.w) + (_e630_ * _e663_.z)));
                            if _e635_ {
                            } else {
                                phi_5505_ = true;
                                break;
                            }
                            buf_b[_e632_] = (_e661_ + _e680_);
                            if _e638_ {
                            } else {
                                phi_5505_ = true;
                                break;
                            }
                            buf_b[_e634_] = (_e661_ - _e680_);
                            phi_5507_ = _e621_;
                        } else {
                            phi_5507_ = _e398_;
                        }
                        let _e684_ = phi_5507_;
                        workgroupBarrier();
                        phi_5506_ = _e684_;
                        phi_2573_ = (_e400_ + 1u);
                    } else {
                        phi_5506_ = _e398_;
                        phi_2573_ = u32();
                    }
                    let _e687_ = phi_5506_;
                    let _e689_ = phi_2573_;
                    phi_5453_ = _e687_;
                    phi_2572_ = _e689_;
                    phi_5505_ = _e687_;
                    if !(_e403_) {
                        break;
                    }
                }
                let _e692_ = phi_5505_;
                if _e692_ {
                    break;
                }
                if _e235_ {
                } else {
                    break;
                }
                let _e694_ = buf_a[_e234_];
                if _e235_ {
                } else {
                    break;
                }
                let _e697_ = buf_b[_e234_].x;
                let _e700_ = buf_b[_e234_].y;
                let _e703_ = buf_b[_e234_].z;
                let _e706_ = buf_b[_e234_].w;
                if _e222_ {
                    let _e707_ = bitcast<u32>(_e211_);
                    let _e708_ = (_e707_ & 2139095040u);
                    if (_e708_ == 2139095040u) {
                        phi_3024_ = true;
                    } else {
                        phi_3024_ = (_e708_ == 0u);
                    }
                    let _e712_ = phi_3024_;
                    phi_3030_ = bitcast<f32>(select(_e707_, 0u, _e712_));
                } else {
                    phi_3030_ = 0f;
                }
                let _e716_ = phi_3030_;
                if _e222_ {
                    let _e717_ = bitcast<u32>(_e215_);
                    let _e718_ = (_e717_ & 2139095040u);
                    if (_e718_ == 2139095040u) {
                        phi_3045_ = true;
                    } else {
                        phi_3045_ = (_e718_ == 0u);
                    }
                    let _e722_ = phi_3045_;
                    phi_3051_ = bitcast<f32>(select(_e717_, 0u, _e722_));
                } else {
                    phi_3051_ = 0f;
                }
                let _e726_ = phi_3051_;
                if _e222_ {
                    let _e727_ = bitcast<u32>(_e219_);
                    let _e728_ = (_e727_ & 2139095040u);
                    if (_e728_ == 2139095040u) {
                        phi_3066_ = true;
                    } else {
                        phi_3066_ = (_e728_ == 0u);
                    }
                    let _e732_ = phi_3066_;
                    phi_3072_ = bitcast<f32>(select(_e727_, 0u, _e732_));
                } else {
                    phi_3072_ = 0f;
                }
                let _e736_ = phi_3072_;
                if _e222_ {
                    let _e748_ = bitcast<u32>(((((_e694_.x * _e694_.x) + (_e694_.y * _e694_.y)) + (_e694_.z * _e694_.z)) + (_e694_.w * _e694_.w)));
                    let _e749_ = (_e748_ & 2139095040u);
                    if (_e749_ == 2139095040u) {
                        phi_3102_ = true;
                    } else {
                        phi_3102_ = (_e749_ == 0u);
                    }
                    let _e753_ = phi_3102_;
                    phi_3108_ = bitcast<f32>(select(_e748_, 0u, _e753_));
                } else {
                    phi_3108_ = 0f;
                }
                let _e757_ = phi_3108_;
                if _e222_ {
                    let _e765_ = bitcast<u32>(((((_e697_ * _e697_) + (_e700_ * _e700_)) + (_e703_ * _e703_)) + (_e706_ * _e706_)));
                    let _e766_ = (_e765_ & 2139095040u);
                    if (_e766_ == 2139095040u) {
                        phi_3130_ = true;
                    } else {
                        phi_3130_ = (_e766_ == 0u);
                    }
                    let _e770_ = phi_3130_;
                    phi_3136_ = bitcast<f32>(select(_e765_, 0u, _e770_));
                } else {
                    phi_3136_ = 0f;
                }
                let _e774_ = phi_3136_;
                let _e776_ = bitcast<u32>((_e757_ * _e716_));
                let _e777_ = (_e776_ & 2139095040u);
                if (_e777_ == 2139095040u) {
                    phi_3148_ = true;
                } else {
                    phi_3148_ = (_e777_ == 0u);
                }
                let _e781_ = phi_3148_;
                let _e2943 = subgroupAdd(bitcast<f32>(select(_e776_, 0u, _e781_)));
                let _e785_ = (_e83_ == 0u);
                if _e235_ {
                } else {
                    break;
                }
                reduce_buf[_e234_] = select(0f, _e2943, _e785_);
                workgroupBarrier();
                phi_3178_ = 128u;
                loop {
                    let _e789_ = phi_3178_;
                    let _e790_ = (_e789_ > 0u);
                    if _e790_ {
                        if (_e82_ < _e789_) {
                            let _e792_ = (_e82_ < 256u);
                            if _e792_ {
                            } else {
                                phi_5521_ = true;
                                break;
                            }
                            let _e794_ = reduce_buf[_e82_];
                            let _e795_ = (_e82_ + _e789_);
                            if (_e795_ < 256u) {
                            } else {
                                phi_5521_ = true;
                                break;
                            }
                            let _e798_ = reduce_buf[_e795_];
                            if _e792_ {
                            } else {
                                phi_5521_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e794_ + _e798_);
                        }
                        workgroupBarrier();
                        phi_3179_ = (_e789_ >> bitcast<u32>(1i));
                    } else {
                        phi_3179_ = u32();
                    }
                    let _e803_ = phi_3179_;
                    phi_3178_ = _e803_;
                    phi_5521_ = _e692_;
                    if !(_e790_) {
                        break;
                    }
                }
                let _e806_ = phi_5521_;
                if _e806_ {
                    break;
                }
                let _e808_ = reduce_buf[0];
                let _e809_ = bitcast<u32>(_e808_);
                let _e810_ = (_e809_ & 2139095040u);
                if (_e810_ == 2139095040u) {
                    phi_3226_ = true;
                } else {
                    phi_3226_ = (_e810_ == 0u);
                }
                let _e814_ = phi_3226_;
                workgroupBarrier();
                let _e818_ = bitcast<u32>((_e774_ * _e716_));
                let _e819_ = (_e818_ & 2139095040u);
                if (_e819_ == 2139095040u) {
                    phi_3243_ = true;
                } else {
                    phi_3243_ = (_e819_ == 0u);
                }
                let _e823_ = phi_3243_;
                let _e3004 = subgroupAdd(bitcast<f32>(select(_e818_, 0u, _e823_)));
                if _e235_ {
                } else {
                    break;
                }
                reduce_buf[_e234_] = select(0f, _e3004, _e785_);
                workgroupBarrier();
                phi_3273_ = 128u;
                loop {
                    let _e829_ = phi_3273_;
                    let _e830_ = (_e829_ > 0u);
                    if _e830_ {
                        if (_e82_ < _e829_) {
                            let _e832_ = (_e82_ < 256u);
                            if _e832_ {
                            } else {
                                phi_5549_ = true;
                                break;
                            }
                            let _e834_ = reduce_buf[_e82_];
                            let _e835_ = (_e82_ + _e829_);
                            if (_e835_ < 256u) {
                            } else {
                                phi_5549_ = true;
                                break;
                            }
                            let _e838_ = reduce_buf[_e835_];
                            if _e832_ {
                            } else {
                                phi_5549_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e834_ + _e838_);
                        }
                        workgroupBarrier();
                        phi_3274_ = (_e829_ >> bitcast<u32>(1i));
                    } else {
                        phi_3274_ = u32();
                    }
                    let _e843_ = phi_3274_;
                    phi_3273_ = _e843_;
                    phi_5549_ = _e806_;
                    if !(_e830_) {
                        break;
                    }
                }
                let _e846_ = phi_5549_;
                if _e846_ {
                    break;
                }
                let _e847_ = reduce_buf[0];
                let _e848_ = bitcast<u32>(_e847_);
                let _e849_ = (_e848_ & 2139095040u);
                if (_e849_ == 2139095040u) {
                    phi_3321_ = true;
                } else {
                    phi_3321_ = (_e849_ == 0u);
                }
                let _e853_ = phi_3321_;
                workgroupBarrier();
                let _e857_ = bitcast<u32>((_e757_ * _e726_));
                let _e858_ = (_e857_ & 2139095040u);
                if (_e858_ == 2139095040u) {
                    phi_3338_ = true;
                } else {
                    phi_3338_ = (_e858_ == 0u);
                }
                let _e862_ = phi_3338_;
                let _e3063 = subgroupAdd(bitcast<f32>(select(_e857_, 0u, _e862_)));
                if _e235_ {
                } else {
                    break;
                }
                reduce_buf[_e234_] = select(0f, _e3063, _e785_);
                workgroupBarrier();
                phi_3368_ = 128u;
                loop {
                    let _e868_ = phi_3368_;
                    let _e869_ = (_e868_ > 0u);
                    if _e869_ {
                        if (_e82_ < _e868_) {
                            let _e871_ = (_e82_ < 256u);
                            if _e871_ {
                            } else {
                                phi_5560_ = true;
                                break;
                            }
                            let _e873_ = reduce_buf[_e82_];
                            let _e874_ = (_e82_ + _e868_);
                            if (_e874_ < 256u) {
                            } else {
                                phi_5560_ = true;
                                break;
                            }
                            let _e877_ = reduce_buf[_e874_];
                            if _e871_ {
                            } else {
                                phi_5560_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e873_ + _e877_);
                        }
                        workgroupBarrier();
                        phi_3369_ = (_e868_ >> bitcast<u32>(1i));
                    } else {
                        phi_3369_ = u32();
                    }
                    let _e882_ = phi_3369_;
                    phi_3368_ = _e882_;
                    phi_5560_ = _e846_;
                    if !(_e869_) {
                        break;
                    }
                }
                let _e885_ = phi_5560_;
                if _e885_ {
                    break;
                }
                let _e886_ = reduce_buf[0];
                let _e887_ = bitcast<u32>(_e886_);
                let _e888_ = (_e887_ & 2139095040u);
                if (_e888_ == 2139095040u) {
                    phi_3416_ = true;
                } else {
                    phi_3416_ = (_e888_ == 0u);
                }
                let _e892_ = phi_3416_;
                workgroupBarrier();
                let _e896_ = bitcast<u32>((_e774_ * _e726_));
                let _e897_ = (_e896_ & 2139095040u);
                if (_e897_ == 2139095040u) {
                    phi_3433_ = true;
                } else {
                    phi_3433_ = (_e897_ == 0u);
                }
                let _e901_ = phi_3433_;
                let _e3122 = subgroupAdd(bitcast<f32>(select(_e896_, 0u, _e901_)));
                if _e235_ {
                } else {
                    break;
                }
                reduce_buf[_e234_] = select(0f, _e3122, _e785_);
                workgroupBarrier();
                phi_3463_ = 128u;
                loop {
                    let _e907_ = phi_3463_;
                    let _e908_ = (_e907_ > 0u);
                    if _e908_ {
                        if (_e82_ < _e907_) {
                            let _e910_ = (_e82_ < 256u);
                            if _e910_ {
                            } else {
                                phi_5571_ = true;
                                break;
                            }
                            let _e912_ = reduce_buf[_e82_];
                            let _e913_ = (_e82_ + _e907_);
                            if (_e913_ < 256u) {
                            } else {
                                phi_5571_ = true;
                                break;
                            }
                            let _e916_ = reduce_buf[_e913_];
                            if _e910_ {
                            } else {
                                phi_5571_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e912_ + _e916_);
                        }
                        workgroupBarrier();
                        phi_3464_ = (_e907_ >> bitcast<u32>(1i));
                    } else {
                        phi_3464_ = u32();
                    }
                    let _e921_ = phi_3464_;
                    phi_3463_ = _e921_;
                    phi_5571_ = _e885_;
                    if !(_e908_) {
                        break;
                    }
                }
                let _e924_ = phi_5571_;
                if _e924_ {
                    break;
                }
                let _e925_ = reduce_buf[0];
                let _e926_ = bitcast<u32>(_e925_);
                let _e927_ = (_e926_ & 2139095040u);
                if (_e927_ == 2139095040u) {
                    phi_3511_ = true;
                } else {
                    phi_3511_ = (_e927_ == 0u);
                }
                let _e931_ = phi_3511_;
                workgroupBarrier();
                let _e935_ = bitcast<u32>((_e757_ * _e736_));
                let _e936_ = (_e935_ & 2139095040u);
                if (_e936_ == 2139095040u) {
                    phi_3528_ = true;
                } else {
                    phi_3528_ = (_e936_ == 0u);
                }
                let _e940_ = phi_3528_;
                let _e3181 = subgroupAdd(bitcast<f32>(select(_e935_, 0u, _e940_)));
                if _e235_ {
                } else {
                    break;
                }
                reduce_buf[_e234_] = select(0f, _e3181, _e785_);
                workgroupBarrier();
                phi_3558_ = 128u;
                loop {
                    let _e946_ = phi_3558_;
                    let _e947_ = (_e946_ > 0u);
                    if _e947_ {
                        if (_e82_ < _e946_) {
                            let _e949_ = (_e82_ < 256u);
                            if _e949_ {
                            } else {
                                phi_5582_ = true;
                                break;
                            }
                            let _e951_ = reduce_buf[_e82_];
                            let _e952_ = (_e82_ + _e946_);
                            if (_e952_ < 256u) {
                            } else {
                                phi_5582_ = true;
                                break;
                            }
                            let _e955_ = reduce_buf[_e952_];
                            if _e949_ {
                            } else {
                                phi_5582_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e951_ + _e955_);
                        }
                        workgroupBarrier();
                        phi_3559_ = (_e946_ >> bitcast<u32>(1i));
                    } else {
                        phi_3559_ = u32();
                    }
                    let _e960_ = phi_3559_;
                    phi_3558_ = _e960_;
                    phi_5582_ = _e924_;
                    if !(_e947_) {
                        break;
                    }
                }
                let _e963_ = phi_5582_;
                if _e963_ {
                    break;
                }
                let _e964_ = reduce_buf[0];
                let _e965_ = bitcast<u32>(_e964_);
                let _e966_ = (_e965_ & 2139095040u);
                if (_e966_ == 2139095040u) {
                    phi_3606_ = true;
                } else {
                    phi_3606_ = (_e966_ == 0u);
                }
                let _e970_ = phi_3606_;
                workgroupBarrier();
                let _e974_ = bitcast<u32>((_e774_ * _e736_));
                let _e975_ = (_e974_ & 2139095040u);
                if (_e975_ == 2139095040u) {
                    phi_3623_ = true;
                } else {
                    phi_3623_ = (_e975_ == 0u);
                }
                let _e979_ = phi_3623_;
                let _e3240 = subgroupAdd(bitcast<f32>(select(_e974_, 0u, _e979_)));
                if _e235_ {
                } else {
                    break;
                }
                reduce_buf[_e234_] = select(0f, _e3240, _e785_);
                workgroupBarrier();
                phi_3653_ = 128u;
                loop {
                    let _e985_ = phi_3653_;
                    let _e986_ = (_e985_ > 0u);
                    if _e986_ {
                        if (_e82_ < _e985_) {
                            let _e988_ = (_e82_ < 256u);
                            if _e988_ {
                            } else {
                                phi_5593_ = true;
                                break;
                            }
                            let _e990_ = reduce_buf[_e82_];
                            let _e991_ = (_e82_ + _e985_);
                            if (_e991_ < 256u) {
                            } else {
                                phi_5593_ = true;
                                break;
                            }
                            let _e994_ = reduce_buf[_e991_];
                            if _e988_ {
                            } else {
                                phi_5593_ = true;
                                break;
                            }
                            reduce_buf[_e82_] = (_e990_ + _e994_);
                        }
                        workgroupBarrier();
                        phi_3654_ = (_e985_ >> bitcast<u32>(1i));
                    } else {
                        phi_3654_ = u32();
                    }
                    let _e999_ = phi_3654_;
                    phi_3653_ = _e999_;
                    phi_5593_ = _e963_;
                    if !(_e986_) {
                        break;
                    }
                }
                let _e1002_ = phi_5593_;
                if _e1002_ {
                    break;
                }
                let _e1003_ = reduce_buf[0];
                let _e1004_ = bitcast<u32>(_e1003_);
                let _e1005_ = (_e1004_ & 2139095040u);
                if (_e1005_ == 2139095040u) {
                    phi_3701_ = true;
                } else {
                    phi_3701_ = (_e1005_ == 0u);
                }
                let _e1009_ = phi_3701_;
                workgroupBarrier();
                let _e1014_ = uniforms.member.member_7_;
                if ((bitcast<u32>(_e1014_) & 2139095040u) != 2139095040u) {
                    phi_3720_ = select(true, false, (_e1014_ > 0.000000000001f));
                } else {
                    phi_3720_ = true;
                }
                let _e1021_ = phi_3720_;
                let _e1022_ = select(_e1014_, 0.000000000001f, _e1021_);
                let _e1025_ = uniforms.member.member_4_;
                let _e1026_ = bitcast<u32>(_e1025_);
                let _e1027_ = (_e1026_ & 2139095040u);
                if (_e1027_ == 2139095040u) {
                    phi_3737_ = true;
                } else {
                    phi_3737_ = (_e1027_ == 0u);
                }
                let _e1031_ = phi_3737_;
                let _e1033_ = bitcast<f32>(select(_e1026_, 0u, _e1031_));
                let _e1036_ = uniforms.member.member_5_;
                let _e1037_ = bitcast<u32>(_e1036_);
                let _e1038_ = (_e1037_ & 2139095040u);
                if (_e1038_ == 2139095040u) {
                    phi_3755_ = true;
                } else {
                    phi_3755_ = (_e1038_ == 0u);
                }
                let _e1042_ = phi_3755_;
                let _e1044_ = bitcast<f32>(select(_e1037_, 0u, _e1042_));
                let _e1047_ = uniforms.member.member_6_;
                let _e1048_ = bitcast<u32>(_e1047_);
                let _e1049_ = (_e1048_ & 2139095040u);
                if (_e1049_ == 2139095040u) {
                    phi_3773_ = true;
                } else {
                    phi_3773_ = (_e1049_ == 0u);
                }
                let _e1053_ = phi_3773_;
                let _e1055_ = bitcast<f32>(select(_e1048_, 0u, _e1053_));
                let _e1056_ = (bitcast<f32>(select(_e848_, 0u, _e853_)) + _e1022_);
                if ((bitcast<u32>(_e1056_) & 2139095040u) != 2139095040u) {
                    let _e1060_ = (_e1056_ <= 0f);
                    if _e1060_ {
                        phi_3872_ = f32();
                    } else {
                        if (_e1056_ < 0.00001f) {
                            let _e1092_ = (select(0.000000000000000000000001f, _e1056_, (_e1056_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e1092_ != _e1092_) {
                                phi_4797_ = true;
                            } else {
                                phi_4797_ = (0.000001f >= _e1092_);
                            }
                            let _e1096_ = phi_4797_;
                            let _e1097_ = select(_e1092_, 0.000001f, _e1096_);
                            let _e1100_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1097_)));
                            let _e1104_ = (_e1100_ + (_e1100_ * (1f - (_e1097_ * _e1100_))));
                            let _e1108_ = (_e1104_ + (_e1104_ * (1f - (_e1097_ * _e1104_))));
                            let _e1114_ = bitcast<u32>(((_e1108_ + (_e1108_ * (1f - (_e1097_ * _e1108_)))) * 18446744000000000000f));
                            let _e1115_ = (_e1114_ & 2139095040u);
                            if (_e1115_ == 2139095040u) {
                                phi_3830_ = true;
                            } else {
                                phi_3830_ = (_e1115_ == 0u);
                            }
                            let _e1119_ = phi_3830_;
                            phi_3871_ = bitcast<f32>(select(_e1114_, 0u, _e1119_));
                        } else {
                            if (_e1056_ != _e1056_) {
                                phi_4737_ = true;
                            } else {
                                phi_4737_ = (0.000001f >= _e1056_);
                            }
                            let _e1065_ = phi_4737_;
                            let _e1066_ = select(_e1056_, 0.000001f, _e1065_);
                            let _e1069_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1066_)));
                            let _e1073_ = (_e1069_ + (_e1069_ * (1f - (_e1066_ * _e1069_))));
                            let _e1077_ = (_e1073_ + (_e1073_ * (1f - (_e1066_ * _e1073_))));
                            let _e1082_ = bitcast<u32>((_e1077_ + (_e1077_ * (1f - (_e1066_ * _e1077_)))));
                            let _e1083_ = (_e1082_ & 2139095040u);
                            if (_e1083_ == 2139095040u) {
                                phi_3865_ = true;
                            } else {
                                phi_3865_ = (_e1083_ == 0u);
                            }
                            let _e1087_ = phi_3865_;
                            phi_3871_ = bitcast<f32>(select(_e1082_, 0u, _e1087_));
                        }
                        let _e1123_ = phi_3871_;
                        phi_3872_ = _e1123_;
                    }
                    let _e1125_ = phi_3872_;
                    phi_3873_ = _e1125_;
                    phi_3874_ = _e1060_;
                } else {
                    phi_3873_ = f32();
                    phi_3874_ = true;
                }
                let _e1127_ = phi_3873_;
                let _e1129_ = phi_3874_;
                let _e1130_ = select(_e1127_, 0f, _e1129_);
                let _e1131_ = (bitcast<f32>(select(_e926_, 0u, _e931_)) + _e1022_);
                if ((bitcast<u32>(_e1131_) & 2139095040u) != 2139095040u) {
                    let _e1135_ = (_e1131_ <= 0f);
                    if _e1135_ {
                        phi_3972_ = f32();
                    } else {
                        if (_e1131_ < 0.00001f) {
                            let _e1167_ = (select(0.000000000000000000000001f, _e1131_, (_e1131_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e1167_ != _e1167_) {
                                phi_4917_ = true;
                            } else {
                                phi_4917_ = (0.000001f >= _e1167_);
                            }
                            let _e1171_ = phi_4917_;
                            let _e1172_ = select(_e1167_, 0.000001f, _e1171_);
                            let _e1175_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1172_)));
                            let _e1179_ = (_e1175_ + (_e1175_ * (1f - (_e1172_ * _e1175_))));
                            let _e1183_ = (_e1179_ + (_e1179_ * (1f - (_e1172_ * _e1179_))));
                            let _e1189_ = bitcast<u32>(((_e1183_ + (_e1183_ * (1f - (_e1172_ * _e1183_)))) * 18446744000000000000f));
                            let _e1190_ = (_e1189_ & 2139095040u);
                            if (_e1190_ == 2139095040u) {
                                phi_3930_ = true;
                            } else {
                                phi_3930_ = (_e1190_ == 0u);
                            }
                            let _e1194_ = phi_3930_;
                            phi_3971_ = bitcast<f32>(select(_e1189_, 0u, _e1194_));
                        } else {
                            if (_e1131_ != _e1131_) {
                                phi_4857_ = true;
                            } else {
                                phi_4857_ = (0.000001f >= _e1131_);
                            }
                            let _e1140_ = phi_4857_;
                            let _e1141_ = select(_e1131_, 0.000001f, _e1140_);
                            let _e1144_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1141_)));
                            let _e1148_ = (_e1144_ + (_e1144_ * (1f - (_e1141_ * _e1144_))));
                            let _e1152_ = (_e1148_ + (_e1148_ * (1f - (_e1141_ * _e1148_))));
                            let _e1157_ = bitcast<u32>((_e1152_ + (_e1152_ * (1f - (_e1141_ * _e1152_)))));
                            let _e1158_ = (_e1157_ & 2139095040u);
                            if (_e1158_ == 2139095040u) {
                                phi_3965_ = true;
                            } else {
                                phi_3965_ = (_e1158_ == 0u);
                            }
                            let _e1162_ = phi_3965_;
                            phi_3971_ = bitcast<f32>(select(_e1157_, 0u, _e1162_));
                        }
                        let _e1198_ = phi_3971_;
                        phi_3972_ = _e1198_;
                    }
                    let _e1200_ = phi_3972_;
                    phi_3973_ = _e1200_;
                    phi_3974_ = _e1135_;
                } else {
                    phi_3973_ = f32();
                    phi_3974_ = true;
                }
                let _e1202_ = phi_3973_;
                let _e1204_ = phi_3974_;
                let _e1205_ = select(_e1202_, 0f, _e1204_);
                let _e1206_ = (bitcast<f32>(select(_e1004_, 0u, _e1009_)) + _e1022_);
                if ((bitcast<u32>(_e1206_) & 2139095040u) != 2139095040u) {
                    let _e1210_ = (_e1206_ <= 0f);
                    if _e1210_ {
                        phi_4072_ = f32();
                    } else {
                        if (_e1206_ < 0.00001f) {
                            let _e1242_ = (select(0.000000000000000000000001f, _e1206_, (_e1206_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                            if (_e1242_ != _e1242_) {
                                phi_5037_ = true;
                            } else {
                                phi_5037_ = (0.000001f >= _e1242_);
                            }
                            let _e1246_ = phi_5037_;
                            let _e1247_ = select(_e1242_, 0.000001f, _e1246_);
                            let _e1250_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1247_)));
                            let _e1254_ = (_e1250_ + (_e1250_ * (1f - (_e1247_ * _e1250_))));
                            let _e1258_ = (_e1254_ + (_e1254_ * (1f - (_e1247_ * _e1254_))));
                            let _e1264_ = bitcast<u32>(((_e1258_ + (_e1258_ * (1f - (_e1247_ * _e1258_)))) * 18446744000000000000f));
                            let _e1265_ = (_e1264_ & 2139095040u);
                            if (_e1265_ == 2139095040u) {
                                phi_4030_ = true;
                            } else {
                                phi_4030_ = (_e1265_ == 0u);
                            }
                            let _e1269_ = phi_4030_;
                            phi_4071_ = bitcast<f32>(select(_e1264_, 0u, _e1269_));
                        } else {
                            if (_e1206_ != _e1206_) {
                                phi_4977_ = true;
                            } else {
                                phi_4977_ = (0.000001f >= _e1206_);
                            }
                            let _e1215_ = phi_4977_;
                            let _e1216_ = select(_e1206_, 0.000001f, _e1215_);
                            let _e1219_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1216_)));
                            let _e1223_ = (_e1219_ + (_e1219_ * (1f - (_e1216_ * _e1219_))));
                            let _e1227_ = (_e1223_ + (_e1223_ * (1f - (_e1216_ * _e1223_))));
                            let _e1232_ = bitcast<u32>((_e1227_ + (_e1227_ * (1f - (_e1216_ * _e1227_)))));
                            let _e1233_ = (_e1232_ & 2139095040u);
                            if (_e1233_ == 2139095040u) {
                                phi_4065_ = true;
                            } else {
                                phi_4065_ = (_e1233_ == 0u);
                            }
                            let _e1237_ = phi_4065_;
                            phi_4071_ = bitcast<f32>(select(_e1232_, 0u, _e1237_));
                        }
                        let _e1273_ = phi_4071_;
                        phi_4072_ = _e1273_;
                    }
                    let _e1275_ = phi_4072_;
                    phi_4073_ = _e1275_;
                    phi_4074_ = _e1210_;
                } else {
                    phi_4073_ = f32();
                    phi_4074_ = true;
                }
                let _e1277_ = phi_4073_;
                let _e1279_ = phi_4074_;
                let _e1280_ = select(_e1277_, 0f, _e1279_);
                let _e1282_ = bitcast<u32>((bitcast<f32>(select(_e809_, 0u, _e814_)) * _e1130_));
                let _e1283_ = (_e1282_ & 2139095040u);
                if (_e1283_ == 2139095040u) {
                    phi_4090_ = true;
                } else {
                    phi_4090_ = (_e1283_ == 0u);
                }
                let _e1287_ = phi_4090_;
                let _e1289_ = bitcast<f32>(select(_e1282_, 0u, _e1287_));
                let _e1291_ = bitcast<u32>((bitcast<f32>(select(_e887_, 0u, _e892_)) * _e1205_));
                let _e1292_ = (_e1291_ & 2139095040u);
                if (_e1292_ == 2139095040u) {
                    phi_4107_ = true;
                } else {
                    phi_4107_ = (_e1292_ == 0u);
                }
                let _e1296_ = phi_4107_;
                let _e1298_ = bitcast<f32>(select(_e1291_, 0u, _e1296_));
                let _e1300_ = bitcast<u32>((bitcast<f32>(select(_e965_, 0u, _e970_)) * _e1280_));
                let _e1301_ = (_e1300_ & 2139095040u);
                if (_e1301_ == 2139095040u) {
                    phi_4124_ = true;
                } else {
                    phi_4124_ = (_e1301_ == 0u);
                }
                let _e1305_ = phi_4124_;
                let _e1307_ = bitcast<f32>(select(_e1300_, 0u, _e1305_));
                let _e1313_ = bitcast<u32>((((_e1033_ * _e1289_) + (_e1044_ * _e1298_)) + (_e1055_ * _e1307_)));
                let _e1314_ = (_e1313_ & 2139095040u);
                if (_e1314_ == 2139095040u) {
                    phi_4145_ = true;
                } else {
                    phi_4145_ = (_e1314_ == 0u);
                }
                let _e1318_ = phi_4145_;
                if (_e82_ == 0u) {
                    if (_e81_.x < _e182_) {
                        if (_e81_.y < _e127_) {
                            let _e1326_ = ((_e81_.y * _e182_) + _e81_.x);
                            if (_e93_ == 0u) {
                                phi_4176_ = 0u;
                            } else {
                                let _e1328_ = (_e93_ - 1u);
                                phi_4176_ = select(_e1328_, _e1326_, (_e1326_ < _e1328_));
                            }
                            let _e1332_ = phi_4176_;
                            if (_e1332_ < _e93_) {
                            } else {
                                break;
                            }
                            band_losses.member[_e1332_] = vec4<f32>(_e1289_, _e1298_, _e1307_, bitcast<f32>(select(_e1313_, 0u, _e1318_)));
                        }
                    }
                }
                if _e222_ {
                    phi_4197_ = (_e694_ * (2f * ((((_e1033_ * _e1130_) * _e716_) + ((_e1044_ * _e1205_) * _e726_)) + ((_e1055_ * _e1280_) * _e736_))));
                } else {
                    phi_4197_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e1347_ = phi_4197_;
                if _e235_ {
                } else {
                    break;
                }
                buf_a[_e234_] = _e1347_;
                workgroupBarrier();
                phi_5610_ = _e1002_;
                phi_4216_ = 0u;
                loop {
                    let _e1349_ = phi_5610_;
                    let _e1351_ = phi_4216_;
                    let _e1353_ = local_8_;
                    let _e1354_ = (_e1351_ < _e1353_);
                    if _e1354_ {
                        let _e1356_ = local_9_;
                        let _e1358_ = ((_e1356_ - 1u) - _e1351_);
                        let _e1359_ = (_e1358_ & 31u);
                        let _e1361_ = (1u << bitcast<u32>(_e1359_));
                        if (_e82_ < _e196_) {
                            let _e1367_ = (_e82_ & (_e1361_ - 1u));
                            let _e1372_ = (((_e82_ >> bitcast<u32>(_e1359_)) << bitcast<u32>(((_e1358_ + 1u) & 31u))) + _e1367_);
                            let _e1373_ = (_e1372_ + _e1361_);
                            let _e1377_ = f32((_e1367_ << bitcast<u32>(((31u - _e1358_) & 31u))));
                            let _e1378_ = (_e1377_ * 0.0000000014629181f);
                            let _e1379_ = bitcast<u32>(_e1378_);
                            if ((_e1379_ & 2139095040u) != 2139095040u) {
                                if (_e1378_ == 0f) {
                                    phi_5692_ = _e1349_;
                                    phi_4472_ = type_27_();
                                    phi_4473_ = true;
                                } else {
                                    let _e1385_ = (bitcast<f32>((_e1379_ & 2147483647u)) > 65536f);
                                    if _e1385_ {
                                        phi_5693_ = _e1349_;
                                        phi_4471_ = type_27_();
                                    } else {
                                        let _e1390_ = (bitcast<i32>(bitcast<u32>(((_e1377_ * 0.0000000009313226f) + 12582912f))) - 1262485504i);
                                        let _e1391_ = f32(_e1390_);
                                        let _e1397_ = (((_e1378_ - (_e1391_ * 1.5703125f)) - (_e1391_ * 0.0004837513f)) - (_e1391_ * 0.0000000754979f));
                                        let _e1398_ = bitcast<u32>(_e1397_);
                                        if ((_e1398_ & 2139095040u) != 2139095040u) {
                                            let _e1403_ = (bitcast<f32>((_e1398_ & 2147483647u)) > 0.85f);
                                            if _e1403_ {
                                                phi_5696_ = _e1349_;
                                                phi_4463_ = type_27_();
                                            } else {
                                                let _e1404_ = (_e1397_ * _e1397_);
                                                let _e1417_ = (_e1397_ + ((_e1397_ * _e1404_) * (-0.16666667f + (_e1404_ * (0.008333334f + (_e1404_ * (-0.0001984127f + (_e1404_ * (0.0000027557319f + (_e1404_ * (-0.000000025052108f + (_e1404_ * 0.00000000016059044f))))))))))));
                                                let _e1430_ = (_e1404_ * (-0.5f + (_e1404_ * (0.041666668f + (_e1404_ * (-0.0013888889f + (_e1404_ * (0.000024801588f + (_e1404_ * (-0.0000002755732f + (_e1404_ * (0.0000000020876756f + (_e1404_ * -0.000000000011470746f)))))))))))));
                                                let _e1431_ = (1f + _e1430_);
                                                switch (_e1390_ & 3i) {
                                                    case 0: {
                                                        phi_4327_ = _e1417_;
                                                        phi_4328_ = _e1431_;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_4327_ = _e1431_;
                                                        phi_4328_ = -(_e1417_);
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_4327_ = -(_e1417_);
                                                        phi_4328_ = (-1f - _e1430_);
                                                        break;
                                                    }
                                                    default: {
                                                        phi_4327_ = (-1f - _e1430_);
                                                        phi_4328_ = _e1417_;
                                                        break;
                                                    }
                                                }
                                                let _e1438_ = phi_4327_;
                                                let _e1440_ = phi_4328_;
                                                let _e1441_ = bitcast<u32>(_e1440_);
                                                let _e1442_ = bitcast<u32>(_e1438_);
                                                if ((_e1441_ & 2147483647u) == 0u) {
                                                    phi_4338_ = true;
                                                } else {
                                                    phi_4338_ = ((_e1442_ & 2147483647u) == 0u);
                                                }
                                                let _e1448_ = phi_4338_;
                                                phi_4341_ = type_32_(0i, 2i);
                                                phi_4344_ = _e1438_;
                                                phi_4346_ = _e1440_;
                                                phi_4348_ = _e1442_;
                                                phi_4350_ = _e1441_;
                                                loop {
                                                    let _e1450_ = phi_4341_;
                                                    let _e1452_ = phi_4344_;
                                                    let _e1454_ = phi_4346_;
                                                    let _e1456_ = phi_4348_;
                                                    let _e1458_ = phi_4350_;
                                                    local_10_ = _e1454_;
                                                    local_11_ = _e1452_;
                                                    if (_e1450_.member < _e1450_.member_1_) {
                                                        let _e1465_ = (_e1450_.member + 1i);
                                                        if select(false, true, ((false == (_e1465_ > _e1450_.member)) != false)) {
                                                            phi_4377_ = type_33_(0u, i32());
                                                        } else {
                                                            phi_4377_ = type_33_(1u, _e1465_);
                                                        }
                                                        let _e1475_ = phi_4377_;
                                                        if (bitcast<i32>(_e1475_.member) != 0i) {
                                                        } else {
                                                            phi_5604_ = true;
                                                            break;
                                                        }
                                                        phi_4342_ = type_32_(_e1475_.member_1_, _e1450_.member_1_);
                                                        phi_4391_ = type_33_(1u, _e1450_.member);
                                                    } else {
                                                        phi_4342_ = _e1450_;
                                                        phi_4391_ = type_33_(0u, i32());
                                                    }
                                                    let _e1485_ = phi_4342_;
                                                    let _e1487_ = phi_4391_;
                                                    if (bitcast<i32>(_e1487_.member) != 0i) {
                                                        let _e1493_ = ((_e1454_ * _e1454_) + (_e1452_ * _e1452_));
                                                        if (_e1493_ > 1f) {
                                                            phi_4415_ = true;
                                                        } else {
                                                            if _e1448_ {
                                                                phi_4413_ = false;
                                                            } else {
                                                                phi_4413_ = (_e1493_ == 1f);
                                                            }
                                                            let _e1497_ = phi_4413_;
                                                            phi_4415_ = _e1497_;
                                                        }
                                                        let _e1499_ = phi_4415_;
                                                        if _e1499_ {
                                                            if ((_e1458_ & 2147483647u) >= (_e1456_ & 2147483647u)) {
                                                                let _e1505_ = (_e1458_ - 1u);
                                                                phi_4430_ = _e1452_;
                                                                phi_4431_ = bitcast<f32>(_e1505_);
                                                                phi_4432_ = _e1456_;
                                                                phi_4433_ = _e1505_;
                                                            } else {
                                                                let _e1503_ = (_e1456_ - 1u);
                                                                phi_4430_ = bitcast<f32>(_e1503_);
                                                                phi_4431_ = _e1454_;
                                                                phi_4432_ = _e1503_;
                                                                phi_4433_ = _e1458_;
                                                            }
                                                            let _e1508_ = phi_4430_;
                                                            let _e1510_ = phi_4431_;
                                                            let _e1512_ = phi_4432_;
                                                            let _e1514_ = phi_4433_;
                                                            phi_4434_ = _e1508_;
                                                            phi_4435_ = _e1510_;
                                                            phi_4436_ = _e1512_;
                                                            phi_4437_ = _e1514_;
                                                        } else {
                                                            phi_4434_ = f32();
                                                            phi_4435_ = f32();
                                                            phi_4436_ = u32();
                                                            phi_4437_ = u32();
                                                        }
                                                        let _e1516_ = phi_4434_;
                                                        let _e1518_ = phi_4435_;
                                                        let _e1520_ = phi_4436_;
                                                        let _e1522_ = phi_4437_;
                                                        phi_4345_ = _e1516_;
                                                        phi_4347_ = _e1518_;
                                                        phi_4349_ = _e1520_;
                                                        phi_4351_ = _e1522_;
                                                        phi_4439_ = _e1499_;
                                                    } else {
                                                        phi_4345_ = f32();
                                                        phi_4347_ = f32();
                                                        phi_4349_ = u32();
                                                        phi_4351_ = u32();
                                                        phi_4439_ = false;
                                                    }
                                                    let _e1524_ = phi_4345_;
                                                    let _e1526_ = phi_4347_;
                                                    let _e1528_ = phi_4349_;
                                                    let _e1530_ = phi_4351_;
                                                    let _e1532_ = phi_4439_;
                                                    phi_4341_ = _e1485_;
                                                    phi_4344_ = _e1524_;
                                                    phi_4346_ = _e1526_;
                                                    phi_4348_ = _e1528_;
                                                    phi_4350_ = _e1530_;
                                                    phi_5604_ = _e1349_;
                                                    if !(_e1532_) {
                                                        break;
                                                    }
                                                }
                                                let _e1535_ = phi_5604_;
                                                phi_5683_ = _e1535_;
                                                if _e1535_ {
                                                    break;
                                                }
                                                let _e1537_ = local_10_;
                                                let _e1538_ = bitcast<u32>(_e1537_);
                                                let _e1545_ = local_11_;
                                                let _e1546_ = bitcast<u32>(_e1545_);
                                                phi_5696_ = _e1535_;
                                                phi_4463_ = type_27_(bitcast<f32>((_e1538_ & select(4294967295u, 2147483648u, ((_e1538_ & 2139095040u) == 0u)))), bitcast<f32>((_e1546_ & select(4294967295u, 2147483648u, ((_e1546_ & 2139095040u) == 0u)))));
                                            }
                                            let _e1554_ = phi_5696_;
                                            let _e1556_ = phi_4463_;
                                            phi_5695_ = _e1554_;
                                            phi_4464_ = _e1556_;
                                            phi_4465_ = _e1403_;
                                        } else {
                                            phi_5695_ = _e1349_;
                                            phi_4464_ = type_27_();
                                            phi_4465_ = true;
                                        }
                                        let _e1558_ = phi_5695_;
                                        let _e1560_ = phi_4464_;
                                        let _e1562_ = phi_4465_;
                                        if _e1562_ {
                                            phi_4470_ = type_27_(1f, 0f);
                                        } else {
                                            phi_4470_ = _e1560_;
                                        }
                                        let _e1564_ = phi_4470_;
                                        phi_5693_ = _e1558_;
                                        phi_4471_ = _e1564_;
                                    }
                                    let _e1566_ = phi_5693_;
                                    let _e1568_ = phi_4471_;
                                    phi_5692_ = _e1566_;
                                    phi_4472_ = _e1568_;
                                    phi_4473_ = _e1385_;
                                }
                                let _e1570_ = phi_5692_;
                                let _e1572_ = phi_4472_;
                                let _e1574_ = phi_4473_;
                                phi_5691_ = _e1570_;
                                phi_4474_ = _e1572_;
                                phi_4475_ = _e1574_;
                            } else {
                                phi_5691_ = _e1349_;
                                phi_4474_ = type_27_();
                                phi_4475_ = true;
                            }
                            let _e1576_ = phi_5691_;
                            let _e1578_ = phi_4474_;
                            let _e1580_ = phi_4475_;
                            if _e1580_ {
                                phi_4480_ = type_27_(1f, 0f);
                            } else {
                                phi_4480_ = _e1578_;
                            }
                            let _e1582_ = phi_4480_;
                            let _e1586_ = select(255u, _e1372_, (_e1372_ < 255u));
                            let _e1588_ = select(255u, _e1373_, (_e1373_ < 255u));
                            let _e1589_ = (_e1586_ < 256u);
                            if _e1589_ {
                            } else {
                                phi_5683_ = true;
                                break;
                            }
                            let _e1591_ = buf_a[_e1586_];
                            let _e1592_ = (_e1588_ < 256u);
                            if _e1592_ {
                            } else {
                                phi_5683_ = true;
                                break;
                            }
                            let _e1594_ = buf_a[_e1588_];
                            if _e1589_ {
                            } else {
                                phi_5683_ = true;
                                break;
                            }
                            buf_a[_e1586_] = (_e1591_ + _e1594_);
                            let _e1598_ = (_e1591_.x - _e1594_.x);
                            let _e1601_ = (_e1591_.y - _e1594_.y);
                            let _e1604_ = (_e1591_.z - _e1594_.z);
                            let _e1607_ = (_e1591_.w - _e1594_.w);
                            if _e1592_ {
                            } else {
                                phi_5683_ = true;
                                break;
                            }
                            buf_a[_e1588_] = vec4<f32>(((_e1582_.member * _e1598_) - (_e1582_.member_1_ * _e1601_)), ((_e1582_.member * _e1601_) + (_e1582_.member_1_ * _e1598_)), ((_e1582_.member * _e1604_) - (_e1582_.member_1_ * _e1607_)), ((_e1582_.member * _e1607_) + (_e1582_.member_1_ * _e1604_)));
                            phi_5685_ = _e1576_;
                        } else {
                            phi_5685_ = _e1349_;
                        }
                        let _e1622_ = phi_5685_;
                        workgroupBarrier();
                        phi_5684_ = _e1622_;
                        phi_4217_ = (_e1351_ + 1u);
                    } else {
                        phi_5684_ = _e1349_;
                        phi_4217_ = u32();
                    }
                    let _e1625_ = phi_5684_;
                    let _e1627_ = phi_4217_;
                    phi_5610_ = _e1625_;
                    phi_4216_ = _e1627_;
                    phi_5683_ = _e1625_;
                    if !(_e1354_) {
                        break;
                    }
                }
                let _e1630_ = phi_5683_;
                if _e1630_ {
                    break;
                }
                if select(_e348_, select(_e348_, false, _e350_), _e358_) {
                    phi_4562_ = 0u;
                    phi_4565_ = 0u;
                    phi_4567_ = _e82_;
                    loop {
                        let _e1632_ = phi_4562_;
                        let _e1634_ = phi_4565_;
                        let _e1636_ = phi_4567_;
                        local_13_ = _e1634_;
                        local_14_ = _e1634_;
                        let _e1638_ = local_12_;
                        let _e1639_ = (_e1632_ < _e1638_);
                        if _e1639_ {
                            phi_4563_ = (_e1632_ + 1u);
                            phi_4566_ = ((_e1634_ << bitcast<u32>(1i)) | (_e1636_ & 1u));
                            phi_4568_ = (_e1636_ >> bitcast<u32>(1i));
                        } else {
                            phi_4563_ = u32();
                            phi_4566_ = u32();
                            phi_4568_ = u32();
                        }
                        let _e1648_ = phi_4563_;
                        let _e1650_ = phi_4566_;
                        let _e1652_ = phi_4568_;
                        phi_4562_ = _e1648_;
                        phi_4565_ = _e1650_;
                        phi_4567_ = _e1652_;
                        if !(_e1639_) {
                            break;
                        }
                    }
                    let _e1655_ = local_13_;
                    let _e1658_ = local_14_;
                    let _e1659_ = select(255u, _e1658_, (_e1655_ < 255u));
                    if (_e1659_ < 256u) {
                    } else {
                        break;
                    }
                    let _e1663_ = buf_a[_e1659_].x;
                    let _e1666_ = buf_a[_e1659_].z;
                    if _e235_ {
                    } else {
                        break;
                    }
                    let _e1667_ = buf_c[_e234_].x;
                    let _e1669_ = bitcast<u32>((_e1663_ * _e1667_));
                    let _e1670_ = (_e1669_ & 2139095040u);
                    if (_e1670_ == 2139095040u) {
                        phi_4641_ = true;
                    } else {
                        phi_4641_ = (_e1670_ == 0u);
                    }
                    let _e1674_ = phi_4641_;
                    let _e1678_ = bitcast<u32>((_e1666_ * _e1667_));
                    let _e1679_ = (_e1678_ & 2139095040u);
                    if (_e1679_ == 2139095040u) {
                        phi_4658_ = true;
                    } else {
                        phi_4658_ = (_e1679_ == 0u);
                    }
                    let _e1683_ = phi_4658_;
                    if (_e91_ == 0u) {
                        phi_4674_ = 0u;
                    } else {
                        let _e1687_ = (_e91_ - 1u);
                        phi_4674_ = select(_e1687_, _e363_, (_e363_ < _e1687_));
                    }
                    let _e1691_ = phi_4674_;
                    if (_e1691_ < _e91_) {
                    } else {
                        break;
                    }
                    d_loss_d_pred.member[_e1691_].x = bitcast<f32>(select(_e1669_, 0u, _e1674_));
                    d_loss_d_pred.member[_e1691_].y = bitcast<f32>(select(_e1678_, 0u, _e1683_));
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn gabor_bezout_grad(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) sg_lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    sg_lid_1_ = sg_lid;
    function_();
    return;
}
